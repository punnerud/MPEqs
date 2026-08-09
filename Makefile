MODEL   ?= models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf
REPO    ?= allenai/OLMoE-1B-7B-0125-Instruct-GGUF
GGUF    ?= OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf
TOKENS  ?= 200000
CHUNK   ?= 1024
# Tokens replayed by fetchbench. Each one reads ~465 MiB uncached, so this is the knob that
# decides whether a sweep takes a minute or twenty.
RTOK    ?= 40
REPS    ?= 5
# Tokens for the contribution trace. Capturing per-expert output norms copies the whole MoE
# output off the device, so this is deliberately two orders of magnitude smaller.
CTOK    ?= 8192
# Tokens for the vector capture. Streams every expert's full output, ~1 MiB per token.
VTOK    ?= 512
KLCHUNKS ?= 24
CACHE_MIB ?= 0,256,512,1024,2048,3072
POLICY_MIB ?= 1024,2048
CHUNK_CHARS ?= 400
EMBED_DIM ?= 384
# Smallest k that keeps the kNN graph connected on this corpus; below it, unreachable pairs
# get capped to one constant and the probe reports structure that is an artefact.
KNN ?= 11
LAYOUT  ?= data/layouts/chain.json
LAYOUTS := data/layouts/identity.json data/layouts/random-1.json \
           data/layouts/random-2.json data/layouts/frequency.json \
           data/layouts/mincut.json data/layouts/chain.json \
           data/layouts/best+greedy.json

CARGO   ?= cargo
BIN     := ./target/release

.PHONY: all fmt clippy test build model corpus trace costmodel layouts fetchbench batch \
        analyze headroom reweight ensemble kfrontier headers project qdepth cache policy study \
        embed probe-embeddings probe-geodesic probe-experts probe-expert-geodesic \
        probe-reinforce sparsemem sparsemem-sanity sparsemem-compression hopcount \
		traversal-cost traversal-verify custom-planted custom-compress \
		custom-arith custom-graphloop custom-ops custom-floor \
		custom-bigmath custom-foreknow custom-exprs custom-equations custom-shrink custom-compression \
		custom-curriculum custom-dreaming custom-experimenting custom-scratchpad custom-fewshot \
        baseline verify verify-routing revert status pipeline bench drop-cache apply \
        check-numbers clean-data help

help: ## Show this help
	@grep -hE '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | sed 's/:.*## /\t/' | sort | column -t -s "$$(printf '\t')"

# check-numbers belongs here: it was a separate target for most of this project's
# life, and "make all passes" was reported as though it covered the documented
# numbers when it covered only the code. One gate, or the gate means less than it
# sounds like.
all: fmt clippy test check-numbers ## Format check, lint, unit tests and the numbers gate

fmt:
	$(CARGO) fmt --all -- --check

clippy:
	$(CARGO) clippy --release --all-targets -- -D warnings

test:
	$(CARGO) test --release
	python3 experiments/loopmem/test_rretl_guard.py 2>&1 | tail -1

build: ## Build all binaries
	$(CARGO) build --release

model: ## Download the model and record its pristine SHA-256
	@mkdir -p models
	hf download $(REPO) $(GGUF) --local-dir models
	$(BIN)/ggufperm init --model $(MODEL)

corpus: ## Fetch the multi-register trace corpus
	./scripts/fetch-corpus.sh

data/trace.bin: | build corpus
	$(BIN)/moetrace --model $(MODEL) --text data/corpus/corpus.txt \
		--out data/trace.bin --max-tokens $(TOKENS) --chunk $(CHUNK)

trace: data/trace.bin ## Capture the router trace with gate weights

data/trace-contrib.bin: | build corpus
	$(BIN)/moetrace --model $(MODEL) --text data/corpus/corpus.txt \
		--out data/trace-contrib.bin --max-tokens $(CTOK) --chunk $(CHUNK) --contrib

analyze: data/trace-contrib.bin ## Per-rank contribution and cross-layer predictability
	$(BIN)/coact analyze --trace data/trace-contrib.bin --out data/analysis.json

headroom: data/trace-contrib.bin ## Ceiling on any router rerank: gate order vs contribution
	$(BIN)/coact headroom --trace data/trace-contrib.bin --out data/headroom.json

data/moe-vecs.bin: | build corpus
	$(BIN)/moetrace --model $(MODEL) --text data/corpus/corpus.txt \
		--out data/trace-vec.bin --vecs data/moe-vecs.bin --vec-tokens $(VTOK) --chunk 512

reweight: data/moe-vecs.bin ## Split truncation damage into scale, selection and recombination
	$(BIN)/coact reweight --trace data/trace-vec.bin --vecs data/moe-vecs.bin \
		--out data/reweight.json

ensemble: data/moe-vecs.bin ## Do the experts vote together, or are they orthogonal?
	$(BIN)/coact ensemble --trace data/trace-vec.bin --vecs data/moe-vecs.bin \
		--out data/ensemble.json

kfrontier: ## Quality against top-k, measured with KL divergence to the unmodified model
	./scripts/kfrontier.sh $(MODEL) $(KLCHUNKS)

headers: ## Range-fetch the GGUF headers of large MoE models (a few MB each)
	./scripts/fetch-headers.sh

project: headers data/costmodel.json ## Per-token fetch cost for models too large to hold
	@set -- ; while read -r h s; do set -- "$$@" "$$h"; done < data/headers/sizes.txt; \
	 hdrs="$$@"; set --; \
	 while read -r h s; do set -- "$$@" "$$s"; done < data/headers/sizes.txt; \
	 $(BIN)/coact project --header $$hdrs --size "$$@" --cost data/costmodel.json \
		--out data/projection.json

qdepth: layouts ## How bandwidth and the layout gain trade off against queue depth
	@for t in 1 2 4 8; do \
		./scripts/drop-cache.sh > /dev/null 2>&1; \
		echo "== queue depth $$t =="; \
		$(BIN)/fetchbench replay --model $(MODEL) --trace data/trace.bin \
			--cost data/costmodel.json --tokens 24 --reps 3 --threads $$t \
			--out data/fetchbench-qd$$t.json \
			--layout data/layouts/identity.json data/layouts/chain.json 2>&1 \
			| grep -E '^(layout|identity|chain|WARNING)'; \
	done

cache: layouts ## LRU expert cache: hit rate and working set against a hard memory budget
	./scripts/drop-cache.sh > /dev/null 2>&1
	$(BIN)/fetchbench cache --model $(MODEL) --trace data/trace.bin \
		--cost data/costmodel.json --tokens 256 --cache-mib $(CACHE_MIB) \
		--layout data/layouts/identity.json data/layouts/chain.json --out data/cache.json

embed: ## Chunk the corpus and embed it (all-minilm via llama-embedding)
	./scripts/embed-corpus.sh $(CHUNK_CHARS) data/embeddings.f32

data/matrix-embeddings.matx: | build
	$(BIN)/matstruct from-embeddings --embeddings data/embeddings.f32 --dim $(EMBED_DIM) \
		--out data/matrix-embeddings.matx

probe-embeddings: data/matrix-embeddings.matx ## Gateway structure of the raw embedding metric
	$(BIN)/matstruct probe --matrix data/matrix-embeddings.matx \
		--out data/matstruct-embeddings.json

probe-geodesic: data/matrix-embeddings.matx ## Same, but as shortest paths over a kNN graph
	$(BIN)/matstruct geodesic --matrix data/matrix-embeddings.matx --k $(KNN) \
		--out data/matrix-geodesic.matx
	$(BIN)/matstruct probe --matrix data/matrix-geodesic.matx \
		--out data/matstruct-geodesic.json

PYTORCH ?= /Users/punnerud/Downloads/ainmt/venv/bin/python3

sparsemem-sanity: ## Validate the sparse-memory probe on a task with a known frontier
	$(PYTORCH) experiments/sparsemem/train.py --task synthetic --slots 4096 \
		--steps 1200 --val-dim 256 --out data/sparsemem/sanity.json

data/labelled/emb.f32: | corpus
	./scripts/embed-labelled.sh 400

sparsemem: data/labelled/emb.f32 ## Is load balancing the cause? Three regimes, one frontier
	$(PYTORCH) experiments/sparsemem/train.py --task embeddings --slots 16384 \
		--steps 1500 --val-dim 1024 --out data/sparsemem/frontier.json

sparsemem-compression: data/labelled/emb.f32 ## How much can a fixed model collapse?
	@echo "fixed model, more data:"; \
	for n in 200 500 1000 2000 4300; do \
		printf '  %-6s ' $$n; \
		$(PYTORCH) experiments/sparsemem/train.py --task embeddings --slots 16384 \
			--steps 1500 --val-dim 1024 --train-subset $$n --only plain \
			--out /dev/null 2>&1 | grep '^plain' | tr -s ' '; \
	done
	@echo "fixed data, smaller net:"; \
	for s in 16384 1024 64 16; do \
		printf '  %-6s ' $$s; \
		$(PYTORCH) experiments/sparsemem/train.py --task embeddings --slots $$s \
			--steps 1500 --val-dim 1024 --only plain --out /dev/null 2>&1 \
			| grep '^plain' | tr -s ' '; \
	done

hopcount: ## Path lengths in the kNN graph — is traversal cheap where the codec failed?
	$(PYTORCH) experiments/hops/hopcount.py

custom-planted: ## Train an MoE with planted structure and score the chain against it
	$(PYTORCH) experiments/customsmoe/planted.py
	python3 experiments/customsmoe/score.py

custom-compress: ## Train routing by compression alone, against plain and load balancing
	$(PYTORCH) experiments/customsmoe/compress.py

custom-arith: ## Exact arithmetic, one pass versus a loop that reuses resident experts
	$(PYTORCH) experiments/customsmoe/arith_loop.py

custom-graphloop: ## Choose each pass from the last by walking the co-activation graph
	$(PYTORCH) experiments/customsmoe/graph_loop.py

custom-exprs: ## Multi-term expressions with brackets: do the hops recover the order?
	$(PYTORCH) experiments/customsmoe/exprs.py

custom-equations: ## Equations with X, a fraction line, and different things each side of =
	$(PYTORCH) experiments/customsmoe/equations.py

landmark: ## MPEE's min-plus landmark model on the layer embeddings (the other matcodec model)
	$(PYTORCH) experiments/customsmoe/dump_layers.py 50000 9600 data/custom/layers
	for t in add mul; do for k in 0 1 2; do \
	  $(BIN)/matstruct landmark -e data/custom/layers/$$t-L$$k.f32 --dim 64 -L 32 \
	    --candidates 8192 --rows 600 -o data/custom/layers/landmark-$$t-L$$k.json; \
	done; done
	$(BIN)/matstruct landmark -m data/matrix-geo11.matx -L 32 --rows 800 \
	  -o data/matstruct-landmark-geo11.json

semantic: ## Real 39-class embeddings: does clustering work where gateways do not?
	$(BIN)/matstruct landmark -e data/labelled/emb.f32 --dim 384 -L 32 --rows 700 \
	  -o data/custom/landmark-labelled.json

visible: ## The control: can the signals find a difficulty axis the input shows?
	$(PYTORCH) experiments/customsmoe/visible.py 9000 data/custom/visible.json 6000,19200

autogroup: ## Can any model-derived signal find the carry partition? (no)
	$(PYTORCH) experiments/customsmoe/autogroup.py 8000 data/custom/autogroup.json \
	  1200,6000,19200

carrygroup: ## Group training data by carry count — the axis the embeddings missed
	$(PYTORCH) experiments/customsmoe/carrygroup.py 6000 6 data/custom/carrygroup6.json \
	  uniform,easy-first,hard-first

trajectory: ## Does landmark structure appear as the network learns? (it does not)
	$(PYTORCH) experiments/customsmoe/trajectory.py 8000 data/custom/traj

arith-minilm: ## The same problems through a pretrained embedder, as a control
	BIN=/opt/homebrew/Cellar/llama.cpp/8500/bin ./scripts/embed-arith.sh 8000 \
	  data/custom/arith-minilm.f32
	$(BIN)/matstruct landmark -e data/custom/arith-minilm.f32 --dim 384 -L 32 --rows 600 \
	  -o data/custom/landmark-arith-minilm.json

custom-fewshot: ## Held-out accuracy against the number of distinct training examples
	$(PYTORCH) experiments/customsmoe/fewshot.py

custom-scratchpad: ## A writable slot memory, and whether it makes multiplication learnable
	$(PYTORCH) experiments/customsmoe/scratchpad.py 8000 data/custom/scratchpad-3x1.json 3 1
	$(PYTORCH) experiments/customsmoe/scratchpad.py 12000 data/custom/scratchpad-3x2.json 3 2

custom-curriculum: ## Let the network choose where to practise, from its own uncertainty
	$(PYTORCH) experiments/customsmoe/curriculum.py

custom-dreaming: ## Read the routing graph backwards to rehearse only what is at risk
	$(PYTORCH) experiments/customsmoe/dreaming.py

custom-experimenting: ## Nudge an operand, check the result, learn from the difference
	$(PYTORCH) experiments/customsmoe/experimenting.py

custom-shrink: ## Shrink the network until reproduced bits exceed stored bits
	$(PYTORCH) experiments/customsmoe/shrink.py

custom-compression: ## One ratio across every network: bits reproduced over bits stored
	python3 experiments/customsmoe/compression_report.py

custom-bigmath: ## Two-digit arithmetic: does harder maths recruit more experts?
	$(PYTORCH) experiments/customsmoe/bigmath.py

custom-foreknow: ## Hop reuse, and whether the needed experts are knowable in advance
	$(PYTORCH) experiments/customsmoe/foreknow.py

custom-floor: ## Compression with a floor that prices lost distinctions
	$(PYTORCH) experiments/customsmoe/floor.py

custom-ops: ## Does the expert split follow the mathematics? (it does not)
	$(PYTORCH) experiments/customsmoe/op_structure.py

traversal-cost: ## Price a walk over an on-disk index against reading the whole index
	$(PYTORCH) experiments/hops/traversal_cost.py

traversal-verify: ## Check that pricing with real uncached reads
	python3 experiments/hops/verify_fetch.py

probe-reinforce: data/trace.bin ## Sweep rich-get-richer reinforcement of the co-activation graph
	@for a in 0 1 2 4 8; do \
		$(BIN)/matstruct from-trace --trace data/trace.bin --layer 8 --reinforce $$a \
			--out data/matrix-rf$$a.matx > /dev/null 2>&1; \
		$(BIN)/matstruct geodesic --matrix data/matrix-rf$$a.matx --k 6 \
			--out data/matrix-rfgeo$$a.matx > /dev/null 2>&1; \
		printf 'alpha=%-3s ' $$a; \
		$(BIN)/matstruct probe --matrix data/matrix-rfgeo$$a.matx \
			--out data/matstruct-rf$$a.json 2>&1 \
			| grep -E 'silhouette|rank-1 explains' | tr -s ' ' | tr '\n' ' '; \
		echo; \
	done

probe-expert-geodesic: data/trace.bin ## The best-structured matrix found: experts, routed
	$(BIN)/matstruct from-trace --trace data/trace.bin --layer 8 \
		--out data/matrix-experts.matx
	$(BIN)/matstruct geodesic --matrix data/matrix-experts.matx --k 6 \
		--out data/matrix-expgeo6.matx
	$(BIN)/matstruct probe --matrix data/matrix-expgeo6.matx \
		--out data/matstruct-expgeo6.json

probe-experts: data/trace.bin ## The negative control: co-activation is not a metric
	$(BIN)/matstruct from-trace --trace data/trace.bin --layer 8 \
		--out data/matrix-experts.matx
	$(BIN)/matstruct probe --matrix data/matrix-experts.matx \
		--out data/matstruct-experts.json

policy: layouts ## Compare cache replacement policies at a fixed memory budget
	./scripts/drop-cache.sh > /dev/null 2>&1
	$(BIN)/fetchbench cache --model $(MODEL) --trace data/trace.bin \
		--cost data/costmodel.json --tokens 96 --cache-mib $(POLICY_MIB) \
		--policy lru,random,hybrid,decayed,static --layout $(LAYOUT) --out data/cache-policy.json

data/costmodel.json: | build
	$(BIN)/fetchbench calibrate --model $(MODEL) --out data/costmodel.json --iters 25

costmodel: data/costmodel.json ## Measure C_fetch and C_byte on this device

layouts: data/trace.bin data/costmodel.json ## Search for expert layouts and score them
	$(BIN)/coact build --model $(MODEL) --trace data/trace.bin \
		--cost data/costmodel.json --outdir data/layouts \
		--report data/layout-report.json --sweeps 6

drop-cache: ## Evict the page cache so fetchbench measures the SSD, not RAM
	./scripts/drop-cache.sh

fetchbench: layouts drop-cache ## Cold-cache replay of every layout
	$(BIN)/fetchbench replay --model $(MODEL) --trace data/trace.bin \
		--cost data/costmodel.json --tokens $(RTOK) --reps $(REPS) \
		--out data/fetchbench.json --layout $(LAYOUTS)

batch: layouts ## Where the gain collapses as requests are batched
	@for b in 1 4 16; do \
		./scripts/drop-cache.sh > /dev/null 2>&1; \
		echo "== batch $$b =="; \
		$(BIN)/fetchbench replay --model $(MODEL) --trace data/trace.bin \
			--cost data/costmodel.json --tokens 48 --batch $$b --reps 3 \
			--out data/fetchbench-b$$b.json \
			--layout data/layouts/identity.json data/layouts/chain.json 2>&1 \
			| grep -E '^(layout|identity|chain|WARNING)'; \
	done

pipeline: trace costmodel layouts fetchbench analyze ## Everything from trace to numbers

study: pipeline headroom reweight ensemble kfrontier project qdepth cache policy ## Every experiment

baseline: ## Record llama-bench and reference logits for the shipped file
	./scripts/baseline.sh $(MODEL)

apply: ## Apply $(LAYOUT) to the model, in place and losslessly
	$(BIN)/ggufperm apply --model $(MODEL) --layout $(LAYOUT)

verify: ## Generations and argmax must be unchanged against the recorded baseline
	$(BIN)/ggufperm verify --model $(MODEL)
	./scripts/verify-lossless.sh $(MODEL)

verify-routing: ## Prove the permutation is exact: layer 0 must route identically
	$(BIN)/moetrace --model $(MODEL) --text data/corpus/corpus.txt \
		--out data/trace-perm4k.bin --max-tokens 4096 --chunk $(CHUNK)
	$(BIN)/ggufperm revert --model $(MODEL)
	$(BIN)/moetrace --model $(MODEL) --text data/corpus/corpus.txt \
		--out data/trace-orig4k.bin --max-tokens 4096 --chunk $(CHUNK)
	$(BIN)/coact compare --base data/trace-orig4k.bin --permuted data/trace-perm4k.bin \
		--layout $(LAYOUT) --out data/routing-agreement.json

bench: ## llama-bench on the current file, warm and with MoE host-side
	llama-bench -m $(MODEL) -r 5 -o json 2>/dev/null > data/bench-current.json
	./scripts/bench-summary.py data/bench-current.json
	llama-bench -m $(MODEL) -r 5 -ncmoe 16 -o json 2>/dev/null > data/bench-current-ncmoe.json
	./scripts/bench-summary.py data/bench-current-ncmoe.json

check-numbers: ## Verify the numbers in the docs still match the data files
	./scripts/check-numbers.py

status:
	$(BIN)/ggufperm status --model $(MODEL)

revert: ## Restore the shipped expert order and re-verify the SHA-256
	$(BIN)/ggufperm revert --model $(MODEL)

clean-data:
	rm -rf data/layouts data/layouts-gate data/logits-* data/*.json data/trace*.bin \
		data/kl data/kl-base.dat data/moe-vecs.bin data/headers data/pressure
