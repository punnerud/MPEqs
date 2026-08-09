#!/usr/bin/env python3
"""Twelve everyday procedures as dependency graphs, for statistical power.

Three tasks could only ever indicate a direction. These twelve are chosen to vary the two things
that should matter: how DEEP the chain is, and how many preconditions JOIN at the widest step.
Several have a join of three or more, which is what the binarisation arm exists to test — the
claim being that a step should always be split to two-into-one, never many-into-one.

Written as (preconditions, effects) so nothing here knows about arithmetic. `hang a picture` is
deliberately the most detailed: eleven actions, a depth-six spine and two independent branches
that must meet, because a short list cannot distinguish a planner from a memoriser.
"""

TASKS = {
    "pour a glass of soda": {
        "goal": "glass full",
        "actions": {
            "open fridge": ([], ["fridge open"]),
            "take out bottle": (["fridge open"], ["bottle on counter"]),
            "unscrew cap": (["bottle on counter"], ["bottle open"]),
            "fetch glass": ([], ["glass on counter"]),
            "pour": (["bottle open", "glass on counter"], ["glass full"]),
        },
    },
    "build a table": {
        "goal": "table standing",
        "actions": {
            "cut the top": ([], ["top cut"]),
            "cut four legs": ([], ["legs cut"]),
            "sand the parts": (["top cut", "legs cut"], ["parts sanded"]),
            "attach legs to top": (["parts sanded"], ["frame assembled"]),
            "stand it up": (["frame assembled"], ["table standing"]),
        },
    },
    "make tea": {
        "goal": "tea ready",
        "actions": {
            "fill kettle": ([], ["kettle full"]),
            "boil water": (["kettle full"], ["water boiled"]),
            "put teabag in cup": ([], ["teabag in cup"]),
            "pour water into cup": (["water boiled", "teabag in cup"], ["tea brewing"]),
            "remove teabag": (["tea brewing"], ["tea ready"]),
        },
    },
    "hang a picture": {
        "goal": "picture hanging straight",
        "actions": {
            "choose the wall spot": ([], ["spot chosen"]),
            "find the stud": (["spot chosen"], ["stud found"]),
            "mark the drill point": (["stud found"], ["point marked"]),
            "fetch the drill": ([], ["drill ready"]),
            "drill the hole": (["point marked", "drill ready"], ["hole drilled"]),
            "push in the wall plug": (["hole drilled"], ["plug seated"]),
            "screw in the hook": (["plug seated"], ["hook mounted"]),
            "unpack the frame": ([], ["frame unpacked"]),
            "fix the wire to the frame": (["frame unpacked"], ["wire fitted"]),
            "hang the frame on the hook": (["hook mounted", "wire fitted"], ["picture hanging"]),
            "level it with the spirit level": (["picture hanging"], ["picture hanging straight"]),
        },
    },
    "make a sandwich": {
        "goal": "sandwich made",
        "actions": {
            "take out the bread": ([], ["bread out"]),
            "cut two slices": (["bread out"], ["slices cut"]),
            "butter the slices": (["slices cut"], ["slices buttered"]),
            "take out the cheese": ([], ["cheese out"]),
            "slice the cheese": (["cheese out"], ["cheese sliced"]),
            "wash a tomato": ([], ["tomato washed"]),
            "slice the tomato": (["tomato washed"], ["tomato sliced"]),
            "lay the filling on the bread":
                (["slices buttered", "cheese sliced", "tomato sliced"], ["filling on"]),
            "close the sandwich": (["filling on"], ["sandwich made"]),
        },
    },
    # The four tasks below exist for the arity question. The first twelve all turned out to join
    # at most two preconditions, so capping the model at two open subgoals changed nothing and
    # both arms returned identical refusal counts on every task — the same experiment run twice.
    # A three- and four-way join is what makes "split to two-into-one" a testable difference.
    "bake a cake": {
        "goal": "cake ready",
        "actions": {
            "weigh the flour": ([], ["flour measured"]),
            "crack the eggs": ([], ["eggs cracked"]),
            "weigh the sugar": ([], ["sugar weighed"]),
            "soften the butter": ([], ["butter soft"]),
            "mix the batter":
                (["flour measured", "eggs cracked", "sugar weighed", "butter soft"],
                 ["batter mixed"]),
            "grease the tin": ([], ["tin greased"]),
            "pour the batter into the tin": (["batter mixed", "tin greased"], ["tin filled"]),
            "heat the oven": ([], ["oven hot"]),
            "put it in the oven": (["tin filled", "oven hot"], ["cake baking"]),
            "take it out after an hour": (["cake baking"], ["cake baked"]),
            "let it cool on a rack": (["cake baked"], ["cake ready"]),
        },
    },
    "make a salad": {
        "goal": "salad served",
        "actions": {
            "wash the lettuce": ([], ["lettuce washed"]),
            "tear the lettuce": (["lettuce washed"], ["lettuce torn"]),
            "chop the cucumber": ([], ["cucumber chopped"]),
            "halve the tomatoes": ([], ["tomatoes halved"]),
            "combine in a bowl":
                (["lettuce torn", "cucumber chopped", "tomatoes halved"], ["bowl filled"]),
            "whisk the dressing": ([], ["dressing made"]),
            "dress the salad": (["bowl filled", "dressing made"], ["salad dressed"]),
            "bring it to the table": (["salad dressed"], ["salad served"]),
        },
    },
    "lay a fire": {
        "goal": "fire burning",
        "actions": {
            "clear out the old ash": ([], ["grate clear"]),
            "crumple some newspaper": ([], ["paper ready"]),
            "split the kindling": ([], ["kindling ready"]),
            "fetch a dry log": ([], ["log ready"]),
            "build the stack":
                (["grate clear", "paper ready", "kindling ready", "log ready"], ["stack built"]),
            "open the flue": ([], ["flue open"]),
            "light the paper": (["stack built", "flue open"], ["fire lit"]),
            "let it draw": (["fire lit"], ["fire burning"]),
        },
    },
    "pack for a trip": {
        "goal": "bag by the door",
        "actions": {
            "check the weather": ([], ["weather known"]),
            "choose the clothes": (["weather known"], ["clothes chosen"]),
            "find the passport": ([], ["passport found"]),
            "charge the phone": ([], ["phone charged"]),
            "print the tickets": ([], ["tickets printed"]),
            "put it all in the bag":
                (["clothes chosen", "passport found", "phone charged", "tickets printed"],
                 ["bag packed"]),
            "weigh the bag": (["bag packed"], ["bag weighed"]),
            "leave it by the door": (["bag weighed"], ["bag by the door"]),
        },
    },
    "do the laundry": {
        "goal": "clothes drying",
        "actions": {
            "sort the clothes": ([], ["clothes sorted"]),
            "load the drum": (["clothes sorted"], ["drum loaded"]),
            "add detergent": (["drum loaded"], ["detergent added"]),
            "close the door": (["detergent added"], ["door closed"]),
            "choose the programme": (["door closed"], ["programme set"]),
            "start the machine": (["programme set"], ["wash running"]),
            "wait for the cycle": (["wash running"], ["wash finished"]),
            "hang the clothes up": (["wash finished"], ["clothes drying"]),
        },
    },
    "change a bicycle tyre": {
        "goal": "wheel back on",
        "actions": {
            "flip the bike over": ([], ["bike upside down"]),
            "release the brake": (["bike upside down"], ["brake released"]),
            "undo the axle nuts": (["brake released"], ["axle free"]),
            "take the wheel off": (["axle free"], ["wheel off"]),
            "lever the tyre off": (["wheel off"], ["tyre off"]),
            "pull out the old tube": (["tyre off"], ["tube out"]),
            "fit the new tube": (["tube out"], ["new tube in"]),
            "seat the tyre back on": (["new tube in"], ["tyre seated"]),
            "pump it up": (["tyre seated"], ["tyre inflated"]),
            "put the wheel back": (["tyre inflated"], ["wheel back on"]),
        },
    },
    "plant a seed in a pot": {
        "goal": "seed planted and watered",
        "actions": {
            "fetch the pot": ([], ["pot ready"]),
            "put gravel in the bottom": (["pot ready"], ["drainage in"]),
            "fill with soil": (["drainage in"], ["soil in"]),
            "make a hole in the soil": (["soil in"], ["hole made"]),
            "take a seed from the packet": ([], ["seed in hand"]),
            "drop the seed in": (["hole made", "seed in hand"], ["seed placed"]),
            "cover it over": (["seed placed"], ["seed covered"]),
            "water it": (["seed covered"], ["seed planted and watered"]),
        },
    },
    "post a letter": {
        "goal": "letter posted",
        "actions": {
            "write the letter": ([], ["letter written"]),
            "fold it": (["letter written"], ["letter folded"]),
            "fetch an envelope": ([], ["envelope ready"]),
            "put the letter in": (["letter folded", "envelope ready"], ["letter inside"]),
            "seal the envelope": (["letter inside"], ["envelope sealed"]),
            "write the address": (["envelope sealed"], ["address written"]),
            "stick on a stamp": (["address written"], ["stamp on"]),
            "drop it in the postbox": (["stamp on"], ["letter posted"]),
        },
    },
    "make french press coffee": {
        "goal": "coffee poured",
        "actions": {
            "boil the water": ([], ["water boiled"]),
            "weigh the beans": ([], ["beans weighed"]),
            "grind the beans": (["beans weighed"], ["coffee ground"]),
            "put grounds in the press": (["coffee ground"], ["grounds in press"]),
            "pour the water in": (["water boiled", "grounds in press"], ["brewing"]),
            "wait four minutes": (["brewing"], ["brew finished"]),
            "push the plunger down": (["brew finished"], ["plunged"]),
            "pour into a cup": (["plunged"], ["coffee poured"]),
        },
    },
    "paint a wall": {
        "goal": "wall painted",
        "actions": {
            "move the furniture out": ([], ["room clear"]),
            "lay down dust sheets": (["room clear"], ["floor covered"]),
            "tape the edges": (["floor covered"], ["edges taped"]),
            "fill the holes": (["edges taped"], ["holes filled"]),
            "sand it smooth": (["holes filled"], ["wall smooth"]),
            "open the paint": ([], ["paint open"]),
            "stir the paint": (["paint open"], ["paint stirred"]),
            "roll on the first coat": (["wall smooth", "paint stirred"], ["first coat on"]),
            "let it dry": (["first coat on"], ["first coat dry"]),
            "roll on the second coat": (["first coat dry"], ["wall painted"]),
        },
    },
    "boil an egg": {
        "goal": "egg ready to eat",
        "actions": {
            "fill a pan with water": ([], ["pan filled"]),
            "put it on the hob": (["pan filled"], ["pan on heat"]),
            "bring it to the boil": (["pan on heat"], ["water boiling"]),
            "take an egg from the box": ([], ["egg in hand"]),
            "lower the egg in": (["water boiling", "egg in hand"], ["egg cooking"]),
            "time seven minutes": (["egg cooking"], ["egg cooked"]),
            "lift it out": (["egg cooked"], ["egg out"]),
            "run it under cold water": (["egg out"], ["egg ready to eat"]),
        },
    },
}


def widest_join(spec):
    return max(len(pre) for pre, _ in spec["actions"].values())


def depth(spec):
    """Longest chain from a precondition-free action to the goal."""
    producers = {}
    for name, (_, eff) in spec["actions"].items():
        for e in eff:
            producers.setdefault(e, []).append(name)
    memo = {}

    def d(name, seen=()):
        if name in seen:
            return 0
        if name in memo:
            return memo[name]
        pre, _ = spec["actions"][name]
        v = 1 + max((d(p2, seen + (name,)) for p in pre for p2 in producers.get(p, [])),
                    default=0)
        memo[name] = v
        return v

    return max(d(n) for n in spec["actions"])


if __name__ == "__main__":
    print(f"{'task':<26}{'actions':>9}{'depth':>7}{'widest join':>13}")
    for name, spec in TASKS.items():
        print(f"{name:<26}{len(spec['actions']):>9}{depth(spec):>7}{widest_join(spec):>13}")
