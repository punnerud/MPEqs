// C shim for capturing MoE router decisions out of llama.cpp.
//
// Why C and not direct Rust FFI: llama_model_params / llama_context_params / ggml_tensor
// are passed and read by value, so any drift between a hand-written Rust struct and the
// installed headers is a silent memory-corruption bug. Compiling against the real headers
// makes the ABI the compiler's problem and leaves Rust with one stable entry point.
//
// llama-debug can already dump tensors, but ggml_print_tensor truncates each dimension to
// three elements — useless when we need all top_k expert ids. Hence the callback.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TOPK_PREFIX "ffn_moe_topk"
#define WEIGHTS_PREFIX "ffn_moe_weights"
#define DOWN_PREFIX "ffn_moe_down"
#define TRACE_MAGIC 0x54454F4Du // "MOET" little-endian
#define TRACE_VERSION 2
#define TRACE_VERSION_CONTRIB 3

typedef struct {
    FILE *   out;
    int32_t  top_k;
    uint64_t n_records;

    // llama.cpp evaluates layers in increasing order within one graph, so a non-increasing
    // layer index means a new ubatch started and the token window advanced.
    int      last_layer;
    uint32_t token_base;
    uint32_t last_ubatch_tokens;

    int32_t  max_tokens;
    int      overflow;

    // ffn_moe_topk and ffn_moe_weights are adjacent nodes in the graph, so the ids are
    // buffered when topk fires and flushed together with the gate weights.
    int      pend_layer;
    int32_t  pend_tokens;
    int32_t *ids;
    size_t   ids_cap;
    void *   raw;
    size_t   raw_cap;
    uint8_t *rec;
    size_t   rec_cap;

    // Optional: L2 norm of each selected expert's pre-gate output, so the *contribution*
    // to the residual can be separated from the router's opinion of it.
    int      want_contrib;
    // Optional companion stream of the raw per-expert output vectors, so truncation and
    // reweighting schemes can be evaluated exactly offline instead of guessed at from norms.
    FILE *   vec_out;
    int32_t  vec_budget;
    uint64_t vec_records;
    int64_t  vec_n_embd;
    void *   vec_buf;
    size_t   vec_buf_cap;
    float   *norms;
    size_t   norms_cap;
    uint16_t *pend_ids;
    size_t   pend_ids_cap;
    float   *pend_w;
    size_t   pend_w_cap;
} trace_ctx;

static void *ensure(void **p, size_t *cap, size_t need) {
    if (need > *cap) {
        void *n = realloc(*p, need);
        if (!n) {
            return NULL;
        }
        *p   = n;
        *cap = need;
    }
    return *p;
}

static int parse_layer(const char *name) {
    const char *dash = strrchr(name, '-');
    if (!dash || !dash[1]) {
        return -1;
    }
    return atoi(dash + 1);
}

static bool trace_cb(struct ggml_tensor *t, bool ask, void *user) {
    trace_ctx *c = (trace_ctx *) user;

    const bool is_topk = strncmp(t->name, TOPK_PREFIX, sizeof(TOPK_PREFIX) - 1) == 0;
    const bool is_wts  = strncmp(t->name, WEIGHTS_PREFIX, sizeof(WEIGHTS_PREFIX) - 1) == 0;
    const bool is_down = c->want_contrib &&
                         strncmp(t->name, DOWN_PREFIX, sizeof(DOWN_PREFIX) - 1) == 0;

    if (ask) {
        return is_topk || is_wts || is_down;
    }
    if (!is_topk && !is_wts && !is_down) {
        return true;
    }

    const int layer = parse_layer(t->name);
    if (layer < 0) {
        fprintf(stderr, "moetrace: cannot parse layer from tensor name '%s'\n", t->name);
        return true;
    }

    if (is_topk) {
        if (t->type != GGML_TYPE_I32) {
            fprintf(stderr, "moetrace: %s is %s, expected I32\n", t->name, ggml_type_name(t->type));
            return true;
        }
        const int32_t k        = (int32_t) t->ne[0];
        const int32_t n_tokens = (int32_t) t->ne[1];
        if (k != c->top_k) {
            fprintf(stderr, "moetrace: %s has ne0=%d, expected top_k=%d\n", t->name, k, c->top_k);
            return true;
        }

        // Layers are evaluated in increasing order within one graph, so a non-increasing
        // index means a new ubatch started and the token window advanced.
        if (layer <= c->last_layer) {
            c->token_base += c->last_ubatch_tokens;
        }
        c->last_layer         = layer;
        c->last_ubatch_tokens = (uint32_t) n_tokens;
        c->pend_layer         = -1;

        if (c->overflow) {
            return true;
        }
        // ffn_moe_topk is a VIEW of the argsort node, not a tensor of its own: llama.cpp
        // builds it as ggml_top_k(probs, n_expert_used), which is ggml_argsort followed by a
        // view keeping the first k of each row. So ne0 == k as expected, but nb[1] is the
        // FULL row stride, n_expert * 4 bytes.
        //
        // A flat ggml_backend_tensor_get(t, dst, 0, ggml_nbytes(t)) therefore copies straight
        // through the gaps and returns each token's entire 64-wide ranking, which the writer
        // then chops into eight bogus "tokens". The signature is unmistakable once looked
        // for: every group of 8 consecutive records was an exact partition of all 64 experts,
        // 25000 groups out of 25000, and the resulting access pattern was uniform by
        // construction — max/min exactly 1.00. Real routing has max/min 13.49.
        //
        // Read row by row instead, and refuse to guess if the geometry is not what we think.
        const size_t row_bytes = (size_t) k * sizeof(int32_t);
        const size_t need      = row_bytes * (size_t) n_tokens;
        if (!ensure((void **) &c->ids, &c->ids_cap, need)) {
            c->overflow = 1;
            return true;
        }
        if (t->nb[0] != sizeof(int32_t)) {
            fprintf(stderr, "moetrace: %s has nb0=%zu, expected %zu\n",
                    t->name, (size_t) t->nb[0], sizeof(int32_t));
            return true;
        }
        for (int32_t i = 0; i < n_tokens; i++) {
            ggml_backend_tensor_get(t, (char *) c->ids + (size_t) i * row_bytes,
                                    (size_t) i * t->nb[1], row_bytes);
        }
        c->pend_layer  = layer;
        c->pend_tokens = n_tokens;
        return true;
    }

    if (is_down) {
        // ffn_moe_down: {n_embd, top_k, n_tokens}, each selected expert's output before
        // the gate weight is applied. Reducing it to an L2 norm here keeps the trace small;
        // copying the raw tensor would be ~64 KiB per token per layer.
        if (c->overflow || !c->want_contrib || c->pend_layer != layer) {
            return true;
        }
        if (t->type != GGML_TYPE_F32) {
            return true;
        }
        const int32_t k = c->top_k;
        const int64_t n_embd = t->ne[0];
        const size_t need = ggml_nbytes(t);
        if (!ensure(&c->raw, &c->raw_cap, need)) {
            c->overflow = 1;
            return true;
        }
        ggml_backend_tensor_get(t, c->raw, 0, need);
        const float *o = (const float *) c->raw;

        const size_t n_pairs = (size_t) k * c->pend_tokens;
        if (!ensure((void **) &c->norms, &c->norms_cap, n_pairs * sizeof(float))) {
            c->overflow = 1;
            return true;
        }
        for (size_t p = 0; p < n_pairs; p++) {
            double acc = 0.0;
            const float *row = o + p * (size_t) n_embd;
            for (int64_t d = 0; d < n_embd; d++) {
                acc += (double) row[d] * (double) row[d];
            }
            c->norms[p] = (float) sqrt(acc);
        }

        if (c->vec_out) {
            c->vec_n_embd = n_embd;
            const size_t need_f16 = (size_t) k * (size_t) n_embd * sizeof(ggml_fp16_t);
            if (!ensure(&c->vec_buf, &c->vec_buf_cap, need_f16)) {
                c->overflow = 1;
                return true;
            }
            for (int32_t i = 0; i < c->pend_tokens; i++) {
                const uint32_t tok = c->token_base + (uint32_t) i;
                if (c->vec_budget > 0 && tok >= (uint32_t) c->vec_budget) {
                    break;
                }
                const float *src = o + (size_t) i * k * (size_t) n_embd;
                ggml_fp32_to_fp16_row(src, (ggml_fp16_t *) c->vec_buf,
                                      (int64_t) k * n_embd);
                if (fwrite(c->vec_buf, 1, need_f16, c->vec_out) != need_f16) {
                    fprintf(stderr, "moetrace: short write to vector stream\n");
                    fclose(c->vec_out);
                    c->vec_out = NULL;
                    break;
                }
                c->vec_records++;
            }
        }

        // Record: u16 layer | u16 pad | u32 token | u16 id*k | f32 weight*k | f32 norm*k
        const size_t rec_len = 8 + (size_t) k * (2 + 4 + 4);
        if (!ensure((void **) &c->rec, &c->rec_cap, rec_len)) {
            c->overflow = 1;
            return true;
        }
        for (int32_t i = 0; i < c->pend_tokens; i++) {
            const uint32_t tok = c->token_base + (uint32_t) i;
            if (c->max_tokens > 0 && tok >= (uint32_t) c->max_tokens) {
                c->overflow = 1;
                break;
            }
            uint16_t lay16 = (uint16_t) layer, zero = 0;
            memcpy(c->rec + 0, &lay16, 2);
            memcpy(c->rec + 2, &zero, 2);
            memcpy(c->rec + 4, &tok, 4);
            for (int32_t j = 0; j < k; j++) {
                const size_t p = (size_t) i * k + j;
                memcpy(c->rec + 8 + 2 * (size_t) j, &c->pend_ids[p], 2);
                memcpy(c->rec + 8 + 2 * (size_t) k + 4 * (size_t) j, &c->pend_w[p], 4);
                memcpy(c->rec + 8 + 6 * (size_t) k + 4 * (size_t) j, &c->norms[p], 4);
            }
            if (fwrite(c->rec, 1, rec_len, c->out) != rec_len) {
                c->overflow = 1;
                break;
            }
            c->n_records++;
        }
        c->pend_layer = -1;
        return true;
    }

    // ffn_moe_weights: shape {1, top_k, n_tokens}, the softmax probability of each
    // selected expert. This is the activation magnitude the expert enters the residual
    // with, and it is what makes a weighted co-activation graph possible.
    if (c->overflow || c->pend_layer != layer) {
        return true;
    }
    if (t->type != GGML_TYPE_F32) {
        fprintf(stderr, "moetrace: %s is %s, expected F32\n", t->name, ggml_type_name(t->type));
        return true;
    }
    const int32_t k = c->top_k;
    const int64_t n_elem = ggml_nelements(t);
    if (n_elem != (int64_t) k * c->pend_tokens) {
        fprintf(stderr, "moetrace: %s has %lld elements, expected %d\n",
                t->name, (long long) n_elem, k * c->pend_tokens);
        return true;
    }
    const size_t need = ggml_nbytes(t);
    if (!ensure(&c->raw, &c->raw_cap, need)) {
        c->overflow = 1;
        return true;
    }
    ggml_backend_tensor_get(t, c->raw, 0, need);
    const float *w = (const float *) c->raw;

    if (c->want_contrib) {
        // Stash ids and weights; the record is written once ffn_moe_down arrives.
        const size_t n = (size_t) k * c->pend_tokens;
        if (!ensure((void **) &c->pend_ids, &c->pend_ids_cap, n * sizeof(uint16_t)) ||
            !ensure((void **) &c->pend_w, &c->pend_w_cap, n * sizeof(float))) {
            c->overflow = 1;
            return true;
        }
        for (size_t i = 0; i < n; i++) {
            c->pend_ids[i] = (uint16_t) c->ids[i];
            c->pend_w[i]   = w[i];
        }
        return true;
    }

    // Record: u16 layer | u16 pad | u32 token | u16 expert_id * k | f32 weight * k
    const size_t rec_len = 8 + (size_t) k * (sizeof(uint16_t) + sizeof(float));
    if (!ensure((void **) &c->rec, &c->rec_cap, rec_len)) {
        c->overflow = 1;
        return true;
    }

    for (int32_t i = 0; i < c->pend_tokens; i++) {
        const uint32_t tok = c->token_base + (uint32_t) i;
        if (c->max_tokens > 0 && tok >= (uint32_t) c->max_tokens) {
            c->overflow = 1;
            break;
        }
        uint16_t lay16 = (uint16_t) layer, zero = 0;
        memcpy(c->rec + 0, &lay16, 2);
        memcpy(c->rec + 2, &zero, 2);
        memcpy(c->rec + 4, &tok, 4);
        for (int32_t j = 0; j < k; j++) {
            uint16_t e = (uint16_t) c->ids[(size_t) i * k + j];
            memcpy(c->rec + 8 + 2 * (size_t) j, &e, 2);
            float wv = w[(size_t) i * k + j];
            memcpy(c->rec + 8 + 2 * (size_t) k + 4 * (size_t) j, &wv, 4);
        }
        if (fwrite(c->rec, 1, rec_len, c->out) != rec_len) {
            fprintf(stderr, "moetrace: short write\n");
            c->overflow = 1;
            break;
        }
        c->n_records++;
    }
    c->pend_layer = -1;
    return true;
}

static char *read_all(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    char *buf = (char *) malloc((size_t) n + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t) n, f) != (size_t) n) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[n] = 0;
    fclose(f);
    *out_len = (size_t) n;
    return buf;
}

// Returns the number of records written, or a negative error code.
int64_t moetrace_run(const char *model_path,
                     const char *text_path,
                     const char *out_path,
                     int32_t     n_gpu_layers,
                     int32_t     n_ctx,
                     int32_t     n_ubatch,
                     int32_t     n_threads,
                     int32_t     n_layer,
                     int32_t     n_expert,
                     int32_t     top_k,
                     int32_t     max_tokens,
                     int32_t     chunk_tokens,
                     int32_t     want_contrib,
                     const char *vec_path,
                     int32_t     vec_budget) {
    int64_t rc      = -1;
    int     decode_failed = 0;

    size_t text_len = 0;
    char * text     = read_all(text_path, &text_len);
    if (!text) {
        fprintf(stderr, "moetrace: cannot read %s\n", text_path);
        return -2;
    }

    // The Metal/BLAS/CPU backends ship as separate dylibs and are discovered at runtime.
    // llama.cpp's own tools do this via common_init(); linking libllama alone is not
    // enough, and without it every model load fails with a bare NULL.
    ggml_backend_load_all();
    llama_backend_init();

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;

    struct llama_model *model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "moetrace: cannot load %s\n", model_path);
        free(text);
        llama_backend_free();
        return -3;
    }

    const struct llama_vocab *vocab = llama_model_get_vocab(model);

    int32_t n_tok_est = -llama_tokenize(vocab, text, (int32_t) text_len, NULL, 0, true, false);
    if (n_tok_est <= 0) {
        fprintf(stderr, "moetrace: tokenizer returned %d\n", n_tok_est);
        goto cleanup_model;
    }
    llama_token *tokens = (llama_token *) malloc((size_t) n_tok_est * sizeof(llama_token));
    if (!tokens) {
        goto cleanup_model;
    }
    int32_t n_tokens = llama_tokenize(vocab, text, (int32_t) text_len, tokens, n_tok_est, true, false);
    if (n_tokens <= 0) {
        fprintf(stderr, "moetrace: tokenize failed (%d)\n", n_tokens);
        free(tokens);
        goto cleanup_model;
    }
    if (max_tokens > 0 && n_tokens > max_tokens) {
        n_tokens = max_tokens;
    }

    FILE *out = fopen(out_path, "wb");
    if (!out) {
        fprintf(stderr, "moetrace: cannot write %s\n", out_path);
        free(tokens);
        goto cleanup_model;
    }

    uint32_t header[6] = { TRACE_MAGIC,
                           want_contrib ? TRACE_VERSION_CONTRIB : TRACE_VERSION, (uint32_t) n_layer, (uint32_t) n_expert, (uint32_t) top_k, 0 };
    uint64_t n_records = 0;
    fwrite(header, sizeof(uint32_t), 6, out);
    fwrite(&n_records, sizeof(uint64_t), 1, out);

    trace_ctx tc;
    memset(&tc, 0, sizeof(tc));
    tc.out        = out;
    tc.top_k      = top_k;
    tc.last_layer = -1;
    tc.pend_layer   = -1;
    tc.max_tokens   = max_tokens;
    tc.want_contrib = want_contrib;
    tc.vec_budget   = vec_budget;
    if (want_contrib && vec_path && vec_path[0]) {
        tc.vec_out = fopen(vec_path, "wb");
        if (!tc.vec_out) {
            fprintf(stderr, "moetrace: cannot write %s\n", vec_path);
        } else {
            // Header patched on close, once n_embd is known from the first tensor.
            uint32_t vh[4] = { 0x564E4F4Du, 1, 0, (uint32_t) top_k };
            uint64_t zero = 0;
            fwrite(vh, sizeof(uint32_t), 4, tc.vec_out);
            fwrite(&zero, sizeof(uint64_t), 1, tc.vec_out);
        }
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = (uint32_t) n_ctx;
    cparams.n_batch           = (uint32_t) n_ubatch;
    cparams.n_ubatch          = (uint32_t) n_ubatch;
    cparams.n_threads         = n_threads;
    cparams.n_threads_batch   = n_threads;
    cparams.cb_eval           = trace_cb;
    cparams.cb_eval_user_data = &tc;
    cparams.no_perf           = true;

    struct llama_context *ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "moetrace: cannot create context\n");
        fclose(out);
        free(tokens);
        goto cleanup_model;
    }

    if (chunk_tokens <= 0 || chunk_tokens > n_ctx) {
        chunk_tokens = n_ctx;
    }

    // Request logits for *every* token. llama.cpp prunes the final layer down to the
    // tokens whose logits are needed, so with the default (last token only) the last MoE
    // layer would be traced for exactly one token per ubatch.
    struct llama_batch batch = llama_batch_init(n_ubatch, 0, 1);

    for (int32_t start = 0; start < n_tokens && !tc.overflow; start += chunk_tokens) {
        // Each chunk is an independent sequence: routing should reflect ordinary context,
        // not an ever-growing one that no real workload would produce.
        llama_memory_clear(llama_get_memory(ctx), true);

        int32_t end = start + chunk_tokens;
        if (end > n_tokens) {
            end = n_tokens;
        }
        for (int32_t p = start; p < end && !tc.overflow; p += n_ubatch) {
            int32_t n = end - p;
            if (n > n_ubatch) {
                n = n_ubatch;
            }
            batch.n_tokens = n;
            for (int32_t i = 0; i < n; i++) {
                batch.token[i]    = tokens[p + i];
                batch.pos[i]      = p - start + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]   = 1;
            }
            int32_t ret = llama_decode(ctx, batch);
            if (ret != 0) {
                fprintf(stderr, "moetrace: llama_decode returned %d at token %d\n", ret, p);
                decode_failed = 1;
                goto cleanup_ctx;
            }
        }
    }

cleanup_ctx:
    // Flush whatever was captured before the failure so the trace file is still readable,
    // but report the failure so the caller does not treat a truncated run as complete.
    rc = decode_failed ? -4 : (int64_t) tc.n_records;
    n_records = tc.n_records;
    fseek(out, 6 * (long) sizeof(uint32_t), SEEK_SET);
    fwrite(&n_records, sizeof(uint64_t), 1, out);
    fclose(out);
    llama_batch_free(batch);
    free(tc.ids);
    free(tc.raw);
    free(tc.rec);
    if (tc.vec_out) {
        uint32_t ne = (uint32_t) tc.vec_n_embd;
        fseek(tc.vec_out, 2 * (long) sizeof(uint32_t), SEEK_SET);
        fwrite(&ne, sizeof(uint32_t), 1, tc.vec_out);
        fseek(tc.vec_out, 4 * (long) sizeof(uint32_t), SEEK_SET);
        fwrite(&tc.vec_records, sizeof(uint64_t), 1, tc.vec_out);
        fclose(tc.vec_out);
        fprintf(stderr, "moetrace: wrote %llu expert-output vectors (n_embd=%u)\n",
                (unsigned long long) tc.vec_records, ne);
    }
    free(tc.vec_buf);
    free(tc.norms);
    free(tc.pend_ids);
    free(tc.pend_w);
    free(tokens);
    llama_free(ctx);

cleanup_model:
    llama_model_free(model);
    llama_backend_free();
    free(text);
    return rc;
}
