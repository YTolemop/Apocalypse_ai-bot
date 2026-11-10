"""階段預備：準備零件、燃油與撤離階段切換。

單檔 OR-Tools 人機，使用原版 apocalypse_sdk 的狀態與行動接口。
呼叫 make_policy()，為每名角色、每場比賽建立獨立策略。
"""
from __future__ import annotations

import heapq
import math
import time
from collections import Counter
from dataclasses import dataclass

from ortools.sat.python import cp_model
from apocalypse_sdk import goto, move, pickup, dropoff, drop, fix, revive, heal, use_medkit, take_medkit, take_weapon, wait

PROFILES = {'anticipator': {'part': 14,
                 'fuel': 65,
                 'civilian': 22,
                 'deadline': 0.3,
                 'engineer': 0.7,
                 'rescue': 65,
                 'risk': 5,
                 'seconds': 1.6,
                 'combo': 2,
                 'region': 0,
                 'stage': 160}}


@dataclass
class Visit:
    kind: str
    node: int
    ref: dict
    weight: int = 0
    service: float = 1.1
    value: float = 0
    release: float = 0
    latest: float = 1e6
    loss: float = 0
    infection: float = 0


class Roads:
    """Physical length and risk are separate; chosen routes emit explicit move steps."""
    def __init__(self, st, risk_weight):
        self.st = st
        self.adj = {n: [] for n in st.nodes}
        self.hazard = {n: 0.0 for n in st.nodes}
        for nid, node in st.nodes.items():
            if node.get("safe"):
                continue
            for z in st.zombies:
                d = math.hypot(node["x"]-z["x"], node["y"]-z["y"])
                self.hazard[nid] += max(0, 1-d/4)
                if nid in z.get("path", [])[:3]:
                    self.hazard[nid] += 0.25
        for e in st.edges:
            if e["blocked"]:
                continue
            a, b = e["a"], e["b"]
            risk = (self.hazard[a]+self.hazard[b])/2
            inf = (st.nodes[a]["infection"]+st.nodes[b]["infection"])/2
            cost = e["length"]*(1+risk_weight*(risk+inf*0.3))
            self.adj[a].append((b, e["length"], cost))
            self.adj[b].append((a, e["length"], cost))
        self.cache = {}

    def route(self, a, b):
        if a not in self.cache:
            dist, prev, physical = {a: 0.0}, {}, {a: 0.0}
            queue = [(0.0, a)]
            while queue:
                d, u = heapq.heappop(queue)
                if d > dist[u]:
                    continue
                for v, length, cost in self.adj[u]:
                    nd = d+cost
                    if nd < dist.get(v, math.inf):
                        dist[v], prev[v], physical[v] = nd, u, physical[u]+length
                        heapq.heappush(queue, (nd, v))
            self.cache[a] = dist, prev, physical
        _, prev, physical = self.cache[a]
        if b not in physical:
            return [], math.inf
        path, at = [], b
        while at != a:
            path.append(at)
            at = prev[at]
        return path[::-1], physical[b]

    def exposure(self, a, b, speed):
        path, _ = self.route(a, b)
        hp = infection = risk = 0.0
        last = a
        for nid in path:
            length = next(l for v, l, _ in self.adj[last] if v == nid)
            secs = length/max(0.5, speed)
            h = (self.hazard[last]+self.hazard[nid])/2
            inf = (self.st.nodes[last]["infection"]+self.st.nodes[nid]["infection"])/2
            outside = (int(not self.st.in_zone(last))+int(not self.st.in_zone(nid)))/2
            hp += secs*(h*1.2+outside*self.st.escape.get("zone_damage", 3))
            infection += secs*(h*0.6+inf*0.25+outside)
            risk += secs*(h+inf*0.4+outside*1.5)
            last = nid
        return hp, infection, risk


class Strategy:
    def __init__(self, profile, solve_seconds=0.25, trace=True, horizon_seconds=65, adaptive=True):
        self.name = profile
        self.w = PROFILES[profile]
        self.solve_seconds = solve_seconds
        self.horizon_seconds = horizon_seconds
        self.adaptive = adaptive
        self.trace = [] if trace else None
        self.counts = Counter()
        self.home_region = None
        self.region_since = -1e6
        self.solves = []
        self.last_decision = {}
        self.last_empty_at = -1e6
        self.last_region_grid = None
        self.planned_at = -1e6
        self.last_bailout = -1e6
        self.idle_diagnostics = []
        self.progress_marker = None
        self.progress_at = 0

    def record(self, st, reason, plan, **details):
        self.last_decision = dict(time=round(st.time, 2), profile=self.name, reason=reason, **details)
        self.counts[reason] += 1
        if self.trace is not None:
            self.trace.append(dict(self.last_decision, plan=plan))
        return plan

    def __call__(self, st):
        me, ex = st.you, st.extraction
        marker = (st.escape["parts"], st.escape["fuel"], st.escape["ready_at"] >= 0)
        if marker != self.progress_marker:
            self.progress_marker, self.progress_at = marker, st.time
        if me["state"] != "active":
            return None
        # Shared emergency behavior can interrupt a running plan; ordinary replans cannot.
        plain = Roads(st, 0)
        _, home_dist = plain.route(st.here, ex)
        eta = home_dist/max(0.5, me["speed"])
        depart = st.escape.get("depart_at", -1)
        if depart >= 0:
            if st.here == ex and me["next_node"] < 0:
                if not st.is_idle and me.get("busy") != "wait":
                    return self.record(st, "board_hold", [wait(10)])
                return None
            remaining = me["plan"][me["plan_index"]:]
            if remaining and all(s["action"] in ("move", "goto", "wait") for s in remaining) and remaining[-1]["node"] == ex:
                return None
            cargo = st.my_cargo()
            discard = [t for t in cargo if t["type"] in ("part", "fuel")]
            if eta+8 > depart-st.time:
                discard = cargo
            steps = [drop(t["id"]) for t in discard]
            if me["medkits"] and (me["hp"] < 60 or me["infection"] > 55):
                steps.append(use_medkit())
            return self.record(st, "board", steps+[goto(ex)])
        critical = me["hp"] < 45 or me["infection"] > 65
        if me["medkits"] and (me["hp"] < 62 or me["infection"] > 48) and me.get("busy") not in ("heal", "use_medkit"):
            return self.record(st, "preventive_medkit", [use_medkit()])
        if critical and me.get("busy") not in ("heal", "use_medkit"):
            if me["medkits"]:
                return self.record(st, "emergency_medkit", [use_medkit()])
            hospitals = st.nodes_of("hospital")
            h = min(hospitals, key=lambda n: plain.route(st.here, n)[1], default=None)
            remaining = me["plan"][me["plan_index"]:]
            if h is not None and not any(s["action"] == "heal" for s in remaining):
                return self.record(st, "emergency_hospital", [goto(h), heal()])
        remaining = me["plan"][me["plan_index"]:]
        job = next((s for s in remaining if s["action"] in ("pickup", "fix", "revive")), None)
        invalid = bool(job and job["action"] == "pickup" and (job["task"] not in st.tasks or
                       st.tasks[job["task"]]["status"] not in ("announced", "open") and st.tasks[job["task"]].get("carrier") != me["id"]))
        invalid |= bool(job and job["action"] == "fix" and not any(i["id"] == job["task"] and i["status"] == "active" for i in st.incidents))
        invalid |= bool(job and job["action"] == "revive" and not any(p["id"] == job["target"] and p["state"] == "downed" for p in st.players))
        # Commit to a still-valid next job. Periodically abandoning a shared generator
        # made every bot assume someone else would repair it, so nobody arrived.
        refresh = invalid and st.time >= me["next_plan_at"] and me.get("busy") in ("moving", "")
        if not st.is_idle and not refresh:
            return None
        useless = [t for t in st.my_cargo() if (t["type"] == "part" and st.escape["ready_at"] >= 0) or (t["type"] == "fuel" and st.escape["fuel"] >= st.escape["fuel_required"])]
        if useless:
            return self.record(st, "drop_useless", [drop(t["id"]) for t in useless])
        if st.time < me["next_plan_at"]:
            return None
        if st.time-self.last_empty_at < 5:
            return None
        roads = Roads(st, 0.12 if self.name != "survivor" else 2.5)
        plan = self.optimize(st, roads)
        if plan is not None:
            self.planned_at = st.time
            return plan
        self.last_empty_at = st.time
        if st.is_idle and len(self.idle_diagnostics) < 160:
            self.idle_diagnostics.append(dict(time=round(st.time, 1), node=st.here, hp=me["hp"], infection=me["infection"],
                                             parts=st.escape["parts"], powered=sum(n["type"] == "factory" and n.get("powered") for n in st.nodes.values()),
                                             tasks=len(st.open_tasks()), solver=self.solves[-1] if self.solves else {}))
        if st.my_cargo():
            path, distance = roads.route(st.here, ex)
            if math.isfinite(distance):
                return self.record(st, "deliver_existing", [move(n) for n in path]+[dropoff(t["id"]) for t in st.my_cargo()])
        if st.escape["ready_at"] >= 0 and st.here != ex and math.isfinite(home_dist):
            return self.record(st, "idle_home", [goto(ex)])
        return None

    def region(self, st, node):
        xs = [n["x"] for n in st.nodes.values()]
        ys = [n["y"] for n in st.nodes.values()]
        n = st.nodes[node]
        return int(n["x"] >= (min(xs)+max(xs))/2)+2*int(n["y"] >= (min(ys)+max(ys))/2)

    def candidates(self, st, roads):
        me, w, ex = st.you, self.w, st.extraction
        cargo = st.my_cargo()
        claimed = {i for p in st.players if p["id"] != me["id"] and p["state"] == "active" for i in p.get("targets", [])}
        useful_parts = st.escape["ready_at"] < 0 and st.escape["parts"] < st.escape["parts_required"]
        cover = Counter(st.tasks[i]["type"] for i in claimed if i in st.tasks)
        def owner_factor(node, ref_id):
            # Predict simultaneous choices from public positions/intentions, without shared memory.
            candidates = []
            for p in st.players:
                if p["state"] != "active" or (p["id"] != me["id"] and (p.get("targets") or p.get("fixing"))):
                    continue
                d = roads.route(p.get("next_node", -1) if p.get("next_node", -1) >= 0 else p["node"], node)[1]
                speed = max(0.5, st.rules["player_speed"]*(1-st.rules["load_speed_penalty"]*p.get("load", 0)))
                tie = ((ref_id*1103+p["id"]*31337)%997)/997
                candidates.append((d/speed+tie*3, p["id"]))
            winner = min(candidates, default=(0, me["id"]))[1]
            return 1.0 if winner == me["id"] else 0.55
        items = []
        for t in st.tasks.values():
            if t["status"] not in ("open", "announced") or not st.can_take(t) or t["id"] in claimed:
                continue
            if t["type"] == "part" and not useful_parts:
                continue
            if t["type"] == "fuel" and (st.escape["ready_at"] < 0 or st.escape["fuel"] >= st.escape["fuel_required"]):
                continue
            if t["weight"]+me["load"] > st.rules["capacity"]:
                continue
            _, distance = roads.route(st.here, t["origin"])
            if not math.isfinite(distance):
                continue
            value = w.get("score", 0.22)*t["reward"]
            if t["type"] == "part":
                value += w["part"]*t["weight"]
            elif t["type"] == "fuel":
                value += w["fuel"]
            else:
                value += w["civilian"]
            if st.time > st.raw["duration"]*0.8 and t["type"] in ("part", "fuel"):
                value += 60
            if st.time > st.escape.get("repair_at", 372)*0.35 and t["type"] == "part":
                expected = min(1, st.time/max(1, st.escape.get("repair_at", 372)))
                deficit = max(0, expected-st.escape["parts"]/max(1, st.escape["parts_required"]))
                value += deficit*90*t["weight"]
            if t["type"] == "fuel":
                value += 100
            value *= owner_factor(t["origin"], t["id"])
            value /= 1+w.get("coverage", 0)*cover[t["type"]]
            latest = t.get("expires_at", -1)
            items.append(Visit(t["type"], t["origin"], t, int(t["weight"]), value=value,
                               release=max(0, t.get("available_at", 0)-st.time),
                               latest=latest-st.time if latest >= 0 else 1e6))
        for inc in st.active_incidents():
            _, distance = roads.route(st.here, inc["node"])
            if not math.isfinite(distance):
                continue
            eta = distance/max(0.5, me["speed"])
            inbound = [p for p in st.players if p["id"] != me["id"] and p["state"] == "active" and inc["id"] in p.get("fixing", [])]
            # Do not crowd an already claimed event unless existing workers cannot finish in time.
            if inbound and inc["deadline"]-st.time > eta+inc["need"]-inc["progress"]+5:
                continue
            remain = inc["need"]-inc["progress"]-inc["workers"]*eta
            if remain <= 0:
                continue
            work = max(0.2, remain/(1+inc["workers"]))+0.2
            if st.time+eta+work+1 >= inc["deadline"]:
                continue
            kind = inc["type"]
            team = 0
            if kind == "generator" and useful_parts:
                # Observed factory count, no access to future spawn schedule.
                facilities = sum(n["type"] == "factory" and n["district"] == st.nodes[inc["node"]]["district"] for n in st.nodes.values())
                team = 90+20*facilities
            elif kind == "bridge":
                team = 55+8*sum(p["state"] == "active" for p in st.players)
            elif kind in ("gas", "heli"):
                team = 80 if st.escape.get("depart_at", -1) < 0 else 0
            elif kind in ("rubble", "chest") and useful_parts:
                team = 45
            elif kind in ("airdrop", "medevac", "napalm", "decoy", "herbs", "guard", "outbreak", "boss"):
                team = 20
            value = inc["reward"]*0.35/(1+inc["workers"])+w["engineer"]*team
            if kind == "generator" and useful_parts and not any(n["type"] == "factory" and n.get("powered") for n in st.nodes.values()):
                value += 480  # Restoring production is a shared requirement for every style.
            if kind == "medevac":
                value += w["rescue"]*sum(p["state"] != "active" for p in st.players)
            if kind == "airdrop":
                value += sum(max(0, 100-p["hp"]) for p in st.players if p["state"] == "active")*0.08
            if kind == "bell":
                value += w["civilian"]*sum(t["type"] == "rescue" and t["status"] in ("open", "announced") for t in st.tasks.values())
            value *= owner_factor(inc["node"], inc["id"]+100000)
            items.append(Visit("event", inc["node"], inc, service=work, value=value, latest=inc["deadline"]-st.time-work-0.5))
        for p in st.players:
            if p["id"] == me["id"] or p["state"] != "downed":
                continue
            value = w["rescue"]+(110 if not st.in_zone(p["node"]) else 0)
            items.append(Visit("revive", p["node"], p, service=3.1, value=value, latest=p["downed_timer"]-4))
        # The same supply choices exist for every profile; high health has no healing benefit.
        if me["hp"] < 80 or me["infection"] > 25:
            for node in st.nodes_of("hospital"):
                v = (100-me["hp"])*1.1+me["infection"]*0.8
                if self.name == "survivor":
                    v *= 1.8
                items.append(Visit("heal", node, {}, service=4.1, value=v))
        if me["medkits"] == 0:
            for node in st.nodes_of("supply"):
                if st.nodes[node].get("medkits", 0) > 0:
                    items.append(Visit("medkit", node, {}, weight=1, value=22 if self.name != "survivor" else 48))
        if me["ammo"] == 0 and st.zombies:
            for node in st.nodes_of("armory"):
                if st.nodes[node].get("weapons", 0) > 0:
                    items.append(Visit("weapon", node, {}, weight=st.rules["weapon_weight"], value=w.get("weapon", 28 if self.name != "survivor" else 45)))
        repair_in = st.escape.get("repair_at", 372)-st.time
        committed = sum(t["weight"] for t in st.tasks.values() if t["type"] == "part" and t["status"] == "carried")
        if st.escape["ready_at"] < 0 and 1 < repair_in <= 35 and st.escape["parts"]+committed >= st.escape["parts_required"]:
            items.append(Visit("stage", ex, {}, service=min(30, repair_in)+0.1, value=w.get("stage", 10)))
        return items

    def optimize(self, st, roads):
        me, w, ex = st.you, self.w, st.extraction
        capacity = st.rules["capacity"]
        horizon = min(self.horizon_seconds, st.raw["duration"]-st.time-2)
        if st.escape.get("depart_at", -1) >= 0:
            horizon = min(horizon, st.escape["depart_at"]-st.time-5)
        if horizon < 1:
            return None
        items = self.candidates(st, roads)
        if self.adaptive:
            # A short planning window must not forbid remote generators and factories entirely.
            primary = [v for v in items if v.kind in ("part", "fuel") or v.kind == "event" and v.ref.get("type") == "generator"]
            trips = []
            for v in primary:
                outbound = roads.route(st.here, v.node)[1]/max(0.5, me["speed"])
                after = max(0.5, st.rules["player_speed"]*(1-st.rules["load_speed_penalty"]*(me["load"]+v.weight)))
                inbound = roads.route(v.node, ex)[1]/after
                if math.isfinite(outbound+inbound):
                    trips.append(max(outbound, v.release)+v.service+inbound+3)
            if trips:
                horizon = min(st.raw["duration"]-st.time-3, max(horizon, min(210, min(trips)+15)))
        # Preserve several action families; pruning is a bounded approximation, not a role filter.
        def rank(v):
            d = roads.route(st.here, v.node)[1]+roads.route(v.node, ex)[1]
            urgent = w["deadline"]*v.ref.get("bonus", 0) if v.ref.get("urgent") else 0
            area = w["region"] if self.home_region is not None and self.region(st, v.node) == self.home_region else 0
            return (v.value+urgent+area)/(d/max(0.5, me["speed"])+v.service+3)
        items = sorted((v for v in items if math.isfinite(roads.route(st.here, v.node)[1]) and math.isfinite(roads.route(v.node, ex)[1])), key=rank, reverse=True)
        chosen = items[:5]
        for family in ("part", "fuel", "rescue", "event", "revive", "heal", "medkit", "weapon", "stage"):
            extra = next((v for v in items if v.kind == family), None)
            if extra is not None and all(extra is not x for x in chosen):
                chosen.append(extra)
        chosen = chosen[:10]
        if not chosen:
            return None
        n = len(chosen)
        visits = [Visit("start", st.here, {})]+chosen+[Visit("end", ex, {})]
        end = n+1
        model = cp_model.CpModel()
        selected = {i: model.new_bool_var(f"take_{i}") for i in range(1, end)}
        limit = math.ceil(horizon*10)
        at = {i: model.new_int_var(0, limit, f"at_{i}") for i in range(end+1)}
        load = {i: model.new_int_var(0, capacity, f"load_{i}") for i in range(end+1)}
        model.add(at[0] == 0)
        model.add(load[0] == me["load"])
        objective = []
        cargo_count = len(st.my_cargo())
        drops = sum(selected[i] for i in selected if visits[i].kind in ("part", "fuel", "rescue"))+cargo_count
        model.add(at[end]+drops*11 <= limit)
        for i, take in selected.items():
            v = visits[i]
            model.add(at[i] >= math.ceil(v.release*10)).only_enforce_if(take)
            model.add(at[i] <= math.floor(v.latest*10)).only_enforce_if(take)
            objective.append(round(v.value*10)*take)
            if v.kind in ("part", "fuel", "rescue"):
                objective.append(round(w["combo"]*10)*take)
            if v.ref.get("urgent"):
                on_time = model.new_bool_var(f"on_time_{i}")
                model.add(on_time <= take)
                model.add(at[end]+drops*11 <= math.floor((v.ref["deadline"]-st.time)*10)).only_enforce_if(on_time)
                objective.append(round(w["deadline"]*v.ref["bonus"]*10)*on_time)
        # A shared anti-stall preference: at least one main-objective job becomes
        # valuable after 45 seconds without progress, even if its net score is modest.
        if st.time-self.progress_at >= 45:
            main = [selected[i] for i in selected if visits[i].kind in ("part", "fuel") or
                    visits[i].kind == "event" and visits[i].ref.get("type") == "generator"]
            if main:
                progresses = model.new_bool_var("break_stall")
                model.add_max_equality(progresses, main)
                objective.append(6000*progresses)
        # Prevent collecting multiple redundant medkits/weapons/heals in the same trip.
        for family in ("medkit", "weapon", "heal"):
            model.add(sum(selected[i] for i in selected if visits[i].kind == family) <= 1)
        if w.get("light"):
            excess = model.new_int_var(0, capacity, "heavy_load")
            model.add(excess >= load[end]-6)
            objective.append(-int(w["light"]*10)*excess)
        if self.name == "regional":
            regions = [model.new_bool_var(f"region_{k}") for k in range(4)]
            model.add_exactly_one(regions)
            for i, take in selected.items():
                r = self.region(st, visits[i].node)
                same = model.new_bool_var(f"same_region_{i}")
                model.add(same <= take)
                model.add(same <= regions[r])
                model.add(same >= take+regions[r]-1)
                crowd = sum(p["state"] == "active" and p["id"] != me["id"] and self.region(st, p["node"]) == r and p["node"] != ex for p in st.players)
                objective.append(round(w["region"]*10)*same-round(crowd*6*10)*take)
            if self.home_region is not None and st.time-self.region_since < 80:
                objective.append(450*regions[self.home_region])
        arcs = [(end, 0, model.new_constant(1))]
        for i, take in selected.items():
            arcs.append((i, i, take.Not()))
        edges = {}
        hp_terms, inf_terms = [], []
        # Physical travel time is tabled for each possible carried weight (not risk-adjusted distance).
        speed_factor = me["speed"]/max(0.5, st.rules["player_speed"]*(1-st.rules["load_speed_penalty"]*me["load"]))
        inverse_table = [math.ceil(1000/max(0.5, st.rules["player_speed"]*(1-st.rules["load_speed_penalty"]*carried)*speed_factor)) for carried in range(capacity+1)]
        inverse = {}
        for i in range(end):
            inverse[i] = model.new_int_var(min(inverse_table), max(inverse_table), f"inverse_speed_{i}")
            model.add_element(load[i], inverse_table, inverse[i])
        for i in range(end):
            for j in range(1, end+1):
                if i == j:
                    continue
                path, distance = roads.route(visits[i].node, visits[j].node)
                if not math.isfinite(distance):
                    continue
                use = model.new_bool_var(f"arc_{i}_{j}")
                edges[i, j] = use
                arcs.append((i, j, use))
                # One inverse-speed lookup per visit, not per arc. Round physical lengths
                # and milliseconds/metre upward, then convert to deciseconds.
                maximum = math.ceil(distance*100)*max(inverse_table)//10000+2+len(path)
                dt = model.new_int_var(0, maximum, f"travel_{i}_{j}")
                moving = model.new_int_var(0, maximum, f"moving_{i}_{j}")
                model.add_division_equality(moving, math.ceil(distance*100)*inverse[i]+9999, 10000)
                model.add(dt == moving+len(path))
                service = 0 if i == 0 else math.ceil(visits[i].service*10)
                model.add(at[j] >= at[i]+service+dt).only_enforce_if(use)
                model.add(load[j] == load[i]+visits[j].weight).only_enforce_if(use)
                active = model.new_int_var(0, maximum, f"active_travel_{i}_{j}")
                model.add(active == dt).only_enforce_if(use)
                model.add(active == 0).only_enforce_if(use.Not())
                hp, infection, risk = roads.exposure(visits[i].node, visits[j].node, 1.0)
                # Edge-average exposure rate times the exact load-dependent travel duration.
                # This removes three redundant element tables per arc without conflating risk and distance.
                denominator = max(0.1, distance)
                hp_terms.append(math.ceil(hp/denominator*10)*active)
                inf_terms.append(math.ceil(infection/denominator*10)*active)
                objective.append(-round(w["risk"]*risk/denominator)*active)
        # Stationary exposure while working at a destination also matters.
        for i, take in selected.items():
            v = visits[i]
            hazard = roads.hazard[v.node]
            outside = 0 if st.in_zone(v.node) else 1
            hp_terms.append(round(v.service*(hazard*1.2+outside*st.escape.get("zone_damage", 3))*100)*take)
            inf_terms.append(round(v.service*(hazard*0.6+outside)*100)*take)
            objective.append(-round(w["risk"]*v.service*(hazard+outside)*10)*take)
        # Conservative current-health bounds; healing is not credited before it actually happens.
        model.add(sum(hp_terms) <= max(0, math.floor((me["hp"]-12)*100)))
        model.add(sum(inf_terms) <= max(0, math.floor((90-me["infection"])*100)))
        model.add_circuit(arcs)
        for (i, j), use in edges.items():
            model.add_hint(use, int(i == 0 and j == end))
        for take in selected.values():
            model.add_hint(take, 0)
        model.maximize(sum(objective)-round(w["seconds"])*at[end])
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.solve_seconds
        solver.parameters.max_deterministic_time = 0.04
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 2026
        start = time.perf_counter()
        status = solver.solve(model)
        self.solves.append(dict(status=solver.status_name(status), seconds=time.perf_counter()-start))
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        if not any(solver.value(v) for v in selected.values()):
            return None
        order, i = [], 0
        while i != end:
            j = next(j for (a, j), use in edges.items() if a == i and solver.value(use))
            order.append(j)
            i = j
        if self.name == "regional":
            region = next(k for k, var in enumerate(regions) if solver.value(var))
            if region != self.home_region:
                self.home_region, self.region_since = region, st.time
        steps, picked, last, details = [], [], st.here, []
        for i in order:
            v = visits[i]
            path, _ = roads.route(last, v.node)
            steps.extend(move(node) for node in path)
            last = v.node
            if i == end:
                break
            if v.kind in ("part", "fuel", "rescue"):
                steps.append(pickup(v.ref["id"]))
                picked.append(v.ref["id"])
            elif v.kind == "event":
                seconds = v.service+0.5
                while seconds > 0:
                    steps.append(fix(v.ref["id"], min(30, seconds)))
                    seconds -= 30
            elif v.kind == "revive":
                steps.append(revive(v.ref["id"]))
            elif v.kind == "heal":
                steps.append(heal())
            elif v.kind == "medkit":
                steps.append(take_medkit())
            elif v.kind == "weapon":
                steps.append(take_weapon())
            elif v.kind == "stage":
                steps.append(wait(min(30, v.service)))
            details.append(dict(kind=v.kind, node=v.node, id=v.ref.get("id"), event=v.ref.get("type") if v.kind == "event" else None, eta=round(solver.value(at[i])/10, 1)))
        for task in st.my_cargo():
            steps.append(dropoff(task["id"]))
        steps.extend(dropoff(task) for task in picked)
        if len(steps) > st.rules["max_plan_steps"]:
            return None
        return self.record(st, "optimized", steps, visits=details, estimated_seconds=round((solver.value(at[end])+11*(len(picked)+cargo_count))/10, 1),
                           objective=solver.objective_value/10, solver_status=solver.status_name(status), region=self.home_region)


BOT_ID = 'anticipator'
BOT_NAME = '階段預備'


def make_policy():
    """建立一個保留跨次狀態的策略實例。"""
    return Strategy(BOT_ID, trace=False)
