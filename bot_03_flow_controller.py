"""產線統籌師 — standalone Apocalypse OR-Tools policy. Game f464379.
Public state only; ordinary plans obey next_plan_at. Create one policy per actor/match."""
from __future__ import annotations
import heapq
import math
import time
from collections import Counter
from dataclasses import dataclass
from ortools.sat.python import cp_model
from apocalypse_sdk import move, goto, pickup, dropoff, fix, revive, heal, use_medkit, take_medkit, take_weapon, drop, wait
PROFILES = {'flow_controller': {'label': '產線統籌師', 'mode': 'cp', 'sets': 1.1, 'time': 1.5, 'risk': 2.0, 'main': 24, 'process': 1.8}}

@dataclass
class Stop:
    node: int
    kind: str
    ref: dict
    delta: int = 0
    service: int = 1
    release: int = 0
    latest: int = 10000
    value: float = 0

class Network:

    def __init__(self, st, profile):
        self.st, self.profile = (st, profile)
        self.adj = {i: [] for i in st.nodes}
        for edge in st.edges:
            if not edge['blocked']:
                a, b, d = (edge['a'], edge['b'], edge['length'])
                self.adj[a].append((b, d))
                self.adj[b].append((a, d))
        self.cache = {}

    def path(self, a, b):
        key = (a, b == self.st.extraction)
        if key not in self.cache:
            dist, prev, queue = ({a: 0.0}, {}, [(0.0, a)])
            while queue:
                cost, u = heapq.heappop(queue)
                if cost != dist[u]:
                    continue
                for v, length in self.adj[u]:
                    if v == self.st.extraction and v != b:
                        continue
                    nd = cost + length
                    if nd < dist.get(v, math.inf):
                        dist[v], prev[v] = (nd, u)
                        heapq.heappush(queue, (nd, v))
            self.cache[key] = (dist, prev)
        dist, prev = self.cache[key]
        if b not in dist:
            return ([], math.inf)
        nodes, at = ([], b)
        while at != a:
            nodes.append(at)
            at = prev[at]
        return (nodes[::-1], dist[b])

    def speed(self, load):
        st, me = (self.st, self.st.you)
        factor = 0.8 if me['hp'] < 30 else 1.0
        if st.time < st.escape.get('speed_boost_until', -1):
            factor *= me['speed'] / max(0.5, st.rules['player_speed'] * (1 - st.rules['load_speed_penalty'] * me['load']) * factor)
        return max(0.5, st.rules['player_speed'] * (1 - st.rules['load_speed_penalty'] * load) * factor)

    def route(self, a, b, depart, load):
        if a == b:
            return ([], depart, 0.0)
        speed = self.speed(load)
        best, clock, previous = ({a: 0.0}, {a: depart}, {})
        queue = [(0.0, a)]
        while queue:
            cost, u = heapq.heappop(queue)
            if cost != best[u]:
                continue
            if u == b:
                break
            for v, length in self.adj[u]:
                if v == self.st.extraction and v != b:
                    continue
                travel = length / speed
                arrival, delay = (clock[u] + travel, 0.0)
                danger = sum((max(0.0, min(arrival + 1.5, y) - max(arrival - 1.0, x)) for x, y in self.st.danger_windows(v)))
                if danger and self.profile.get('safe') and self.st.nodes[u].get('safe'):
                    until = max((y for x, y in self.st.danger_windows(v) if x <= arrival + 1.5 and y >= arrival - 1), default=arrival)
                    if until - arrival < 15:
                        delay = until - arrival + 2
                        arrival += delay
                        danger = 0
                outside = (int(not self.st.in_zone(u)) + int(not self.st.in_zone(v))) / 2
                risk = danger * 4 + travel * (outside * 1.6 + self.st.nodes[v]['infection'] * 0.2)
                nd = cost + travel + delay + self.profile['risk'] * risk
                if nd < best.get(v, math.inf):
                    best[v], clock[v], previous[v] = (nd, arrival, (u, delay))
                    heapq.heappush(queue, (nd, v))
        if b not in clock:
            return ([], math.inf, math.inf)
        chunks, at = ([], b)
        while at != a:
            before, delay = previous[at]
            chunks.append(([wait(delay)] if delay > 0.05 else []) + [move(at)])
            at = before
        return ([step for chunk in reversed(chunks) for step in chunk], clock[b], best[b] - (clock[b] - depart))

class Policy:

    def __init__(self, profile, solve_seconds=0.3):
        self.name = profile
        self.cfg = PROFILES[profile]
        self.solve_seconds = solve_seconds
        self.calls = Counter()
        self.solver_history = []
        self.last_decision = {}
        self.last_solve = -1000.0
        self.planned_tasks = set()

    def __call__(self, st):
        me = st.you
        if st.status != 'running' or me.get('state') != 'active':
            return None
        graph = Network(st, self.cfg)
        ex = st.extraction
        cargo = st.my_cargo()
        useless = [t for t in cargo if t['type'] in ('part', 'delivery') and st.escape['ready_at'] >= 0 or (t['type'] == 'fuel' and st.escape['fuel'] >= st.escape['fuel_required'])]
        if useless:
            return [drop(t['id']) for t in useless]
        depart = st.escape.get('depart_at', -1)
        if depart >= 0:
            if st.here == ex and me['next_node'] < 0:
                return None
            tail = me['plan'][me['plan_index']:]
            if tail and all((s['action'] in ('move', 'goto', 'wait') for s in tail)) and (tail[-1].get('node') == ex):
                return None
            return [drop(t['id']) for t in cargo] + [goto(ex)]
        if me.get('medkits', 0) and (me['hp'] < 80 or me['infection'] > 35) and (me['busy'] not in ('heal', 'use_medkit')):
            return [use_medkit()]
        if (me['hp'] < 68 or me['infection'] > 50) and me['busy'] not in ('heal', 'use_medkit'):
            hospitals = st.nodes_of('hospital')
            target = min(hospitals, key=lambda n: graph.path(st.here, n)[1], default=None)
            tail = me['plan'][me['plan_index']:]
            if target is not None and (not any((s['action'] == 'heal' for s in tail))):
                return [goto(target), heal()]
        if st.time < me['next_plan_at'] or st.time - self.last_solve < 2:
            return None
        if not st.is_idle:
            tail = me['plan'][me['plan_index']:]
            next_job = next((s for s in tail if s['action'] in ('pickup', 'fix', 'revive')), None)
            invalid = False
            if next_job and next_job['action'] == 'pickup':
                task = st.tasks.get(next_job['task'])
                invalid = task is None or (task['status'] == 'carried' and task.get('carrier') != me['id'])
            elif next_job and next_job['action'] == 'fix':
                invalid = not any((i['id'] == next_job['task'] and i['status'] == 'active' for i in st.incidents))
            elif next_job:
                invalid = not any((p['id'] == next_job['target'] and p['state'] == 'downed' for p in st.players))
            if not invalid or me['busy'] not in ('moving', ''):
                return None
        self.last_solve = st.time
        problem = Problem(st, graph, self.cfg)
        if len(problem.stops) <= 2:
            self.last_decision = dict(reason='no_candidates', time=st.time)
            return None
        mode = self.cfg['mode']
        result = self.solve_cp(problem)
        if not result:
            self.last_decision = dict(reason='no_solution', time=st.time)
            return None
        steps, evaluation = problem.to_plan(result)
        self.last_decision = dict(time=st.time, backend=mode, order=result, **evaluation)
        if steps:
            self.calls['plans'] += 1
            self.planned_tasks = {s['task'] for s in steps if s['action'] == 'pickup'}
        return steps or None

    def solve_cp(self, p):
        started = time.perf_counter()
        m = cp_model.CpModel()
        n, horizon = (len(p.stops), p.horizon)
        chosen = {i: m.new_bool_var(f'visit_{i}') for i in range(2, n)}
        t = [m.new_int_var(0, horizon, f't_{i}') for i in range(n)]
        load = [m.new_int_var(0, p.capacity, f'load_{i}') for i in range(n)]
        inverse = [m.new_int_var(1, 2000, f'inverse_speed_{i}') for i in range(n)]
        speeds = [math.ceil(1000 / p.graph.speed(k)) for k in range(p.capacity + 1)]
        for i in range(n):
            m.add_element(load[i], speeds, inverse[i])
        m.add(t[0] == 0)
        m.add(load[0] == p.initial_load)
        arcs, arcvars, objective = ([(1, 0, m.new_constant(1))], {}, [])
        for i in range(2, n):
            arcs.append((i, i, chosen[i].Not()))
        for i in range(n):
            if i == 1:
                continue
            for j in range(1, n):
                if i == j or not math.isfinite(p.distance[i][j]):
                    continue
                v = m.new_bool_var(f'a_{i}_{j}')
                arcs.append((i, j, v))
                arcvars[i, j] = v
                m.add(1000 * t[j] >= 1000 * (t[i] + p.stops[i].service) + math.ceil(p.distance[i][j]) * inverse[i]).only_enforce_if(v)
                m.add(load[j] == load[i] + p.stops[j].delta).only_enforce_if(v)
                objective.append(-round(p.risk[i][j] * p.cfg['risk']) * v)
        m.add_circuit(arcs)
        for i, stop in enumerate(p.stops[2:], 2):
            x = chosen[i]
            m.add(t[i] >= stop.release).only_enforce_if(x)
            m.add(t[i] + stop.service <= min(horizon, stop.latest)).only_enforce_if(x)
            if stop.kind == 'event':
                objective.append(round(stop.value) * x)
            if p.cfg.get('safe'):
                for a, b in p.windows(i):
                    before = m.new_bool_var(f'safe_{i}_{a}')
                    m.add(t[i] + stop.service <= a).only_enforce_if([x, before])
                    m.add(t[i] >= b).only_enforce_if([x, before.Not()])
        for pk, dr in p.pairs:
            m.add(chosen[pk] == chosen[dr])
            m.add(t[dr] >= t[pk] + 1).only_enforce_if(chosen[pk])
        for i in p.must:
            m.add(chosen[i] == 1)
        for parent, child, delay in p.after:
            m.add(chosen[child] <= chosen[parent])
            m.add(t[child] >= t[parent] + p.stops[parent].service + delay).only_enforce_if(chosen[child])
        for dr in p.home_drops:
            for i in range(2, n):
                if p.stops[i].node != p.st.extraction:
                    m.add(t[i] + p.stops[i].service <= t[dr]).only_enforce_if([chosen[i], chosen[dr]])
        for dr in p.drops:
            stop = p.stops[dr]
            task = stop.ref
            base, floor, rate, due = p.reward_terms(task)
            late = m.new_int_var(0, horizon + 1000, f'late_{dr}')
            m.add_max_equality(late, [0, t[dr] + stop.service - due])
            value = m.new_int_var(floor, base, f'value_{dr}')
            m.add_max_equality(value, [floor, base - rate * late])
            earned = m.new_int_var(0, base, f'earned_{dr}')
            m.add(earned == value).only_enforce_if(chosen[dr])
            m.add(earned == 0).only_enforce_if(chosen[dr].Not())
            objective.append(earned)
        for group, bonus in p.sets:
            complete = m.new_bool_var('set_' + str(group[0]))
            for pk in group:
                m.add(complete <= chosen[pk])
            m.add(complete >= sum((chosen[k] for k in group)) - len(group) + 1)
            objective.append(round(bonus * p.cfg['sets']) * complete)
            for pk in group:
                for dr in p.home_drops:
                    m.add(t[pk] + 1 <= t[dr]).only_enforce_if([complete, chosen[dr]])
        objective.append(-round(p.cfg['time']) * t[1])
        m.maximize(sum(objective))
        seed = p.seed_order()
        try:
            selected_seed, times_seed, loads_seed = p.hint_values(seed)
        except ValueError:
            seed, selected_seed, times_seed, loads_seed = ([], set(), {}, {})
        chain = [0] + seed + [1]
        seed_arcs = set(zip(chain, chain[1:]))
        for (i, j), v in arcvars.items():
            m.add_hint(v, int((i, j) in seed_arcs))
        for i, v in chosen.items():
            m.add_hint(v, int(i in selected_seed))
        for i in range(n):
            m.add_hint(t[i], times_seed.get(i, 0))
            m.add_hint(load[i], loads_seed.get(i, p.initial_load))
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = max(0.25, self.solve_seconds)
        solver.parameters.num_search_workers = 1
        solver.parameters.cp_model_presolve = False
        solver.parameters.search_branching = cp_model.HINT_SEARCH
        solver.parameters.random_seed = int(p.st.you['id']) + 17
        status = solver.solve(m)
        self.calls['cp'] += 1
        self.solver_history.append(dict(engine='cp', status=solver.status_name(status), seconds=time.perf_counter() - started))
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None
        result, at = ([], 0)
        while at != 1:
            nxt = next((j for (i, j), v in arcvars.items() if i == at and solver.value(v)), None)
            if nxt is None:
                return None
            at = nxt
            if at != 1:
                result.append(at)
        return result

class Problem:

    def __init__(self, st, graph, cfg):
        self.st, self.graph, self.cfg = (st, graph, cfg)
        self.capacity = min(int(st.rules['capacity']), max(int(st.you['load']), int(cfg.get('max_load', st.rules['capacity']))))
        self.initial_load = int(st.you['load'])
        self.horizon = min(125, max(20, int(st.raw['duration'] - st.time - 5)))
        self.stops = [Stop(st.here, 'start', {}, service=0), Stop(st.extraction, 'end', {}, service=0)]
        self.pairs, self.must, self.after, self.drops, self.sets = ([], [], [], [], [])
        self.pick_to_drop = {}
        self.task_stops = {}
        claimed = st.claimed_tasks()
        self.own_targets = {s['task'] for s in st.you['plan'][st.you['plan_index']:] if s['action'] == 'pickup'}
        cargo = st.my_cargo()
        useful = lambda t: not (t['type'] in ('part', 'delivery') and st.escape['ready_at'] >= 0 or (t['type'] == 'fuel' and st.escape['fuel'] >= st.escape['fuel_required']))
        pool = [t for t in st.tasks.values() if t['status'] in ('open', 'announced') and st.can_take(t) and useful(t) and (t['available_at'] - st.time < self.horizon - 5) and (t.get('expires_at', -1) < 0 or t['expires_at'] > max(st.time, t['available_at']) + 2) and (t['weight'] <= self.capacity) and math.isfinite(graph.path(st.here, t['origin'])[1])]

        def priority(t):
            d = graph.path(st.here, t['origin'])[1] + graph.path(t['origin'], t['dest'])[1]
            reward = t['reward'] + cfg['main'] * t['weight'] * (t['type'] == 'part')
            reward += t.get('set_bonus', 0) / max(1, t.get('set_size', 1)) * cfg['sets'] * 0.4
            if t['type'] == 'delivery':
                reward *= cfg['process']
            if t['id'] in claimed:
                reward *= 0.25
            reward *= 0.9 + 0.2 * ((t['id'] * 73 + st.you['id'] * 131) % 97) / 97
            return reward / (d / max(0.5, graph.speed(min(10, self.initial_load + 4))) + 3)
        limit = 4 if cfg['mode'] in ('cp', 'hybrid') or cfg.get('light') else 6
        chosen = sorted(pool, key=priority, reverse=True)[:limit]
        for task in list(chosen):
            members = st.set_of(task) if task.get('set', -1) >= 0 else []
            if len(members) == task.get('set_size') and len(chosen) + len(members) <= 10:
                chosen += [t for t in members if t in pool and t not in chosen]
        follow = {t['after']: t for t in st.tasks.values() if t.get('status') == 'locked' and t.get('after', -1) >= 0}
        for task in list(chosen) + cargo:
            if task['id'] in follow and useful(follow[task['id']]) and (len(chosen) < 10):
                chosen.append(follow[task['id']])
        for task in chosen:
            if task['id'] in self.task_stops:
                continue
            release = max(0, math.ceil(task['available_at'] - st.time))
            expiry = task.get('expires_at', -1)
            latest = self.horizon if expiry < 0 else max(0, math.floor(expiry - st.time))
            pk = self.add(Stop(task['origin'], 'pick', task, int(task['weight']), release=release, latest=latest))
            dr = self.add(Stop(task['dest'], 'drop', task, -int(task['weight'])))
            self.pairs.append((pk, dr))
            self.pick_to_drop[pk] = dr
            self.task_stops[task['id']] = (pk, dr)
            self.drops.append(dr)
        for task in cargo:
            dr = self.add(Stop(task['dest'], 'drop', task, -int(task['weight'])))
            self.must.append(dr)
            self.drops.append(dr)
            self.task_stops[task['id']] = (None, dr)
        for task in chosen:
            parent = task.get('after', -1)
            if task['status'] == 'locked' and parent in self.task_stops:
                self.after.append((self.task_stops[parent][1], self.task_stops[task['id']][0], math.ceil(task.get('proc_delay', 0))))
        groups = {}
        for pk, dr in self.pairs:
            task = self.stops[pk].ref
            if task.get('set', -1) >= 0:
                groups.setdefault(task['set'], []).append(pk)
        for group in groups.values():
            task = self.stops[group[0]].ref
            carried_members = [t for t in cargo if t.get('set', -1) == task['set']]
            if len(group) + len(carried_members) == task.get('set_size') and sum((self.stops[i].delta for i in group)) + sum((t['weight'] for t in carried_members)) <= self.capacity - self.ballast():
                self.sets.append((group, task['set_bonus']))
        self.add_events()
        self.home_drops = [i for i in self.drops if self.stops[i].node == st.extraction]
        estimate = min(self.capacity - 1, max(self.initial_load, 6 if cfg.get('light') else 12))
        speed = graph.speed(estimate)
        self.distance = [[graph.path(a.node, b.node)[1] for b in self.stops] for a in self.stops]
        self.travel = [[min(10000, math.ceil(d / speed)) if math.isfinite(d) else 10000 for d in row] for row in self.distance]
        self.risk = [[(self.stops[i].node != self.stops[j].node) * self.travel[i][j] * (st.nodes[self.stops[j].node]['infection'] * 0.25 + 0.7 * int(not st.in_zone(self.stops[j].node))) for j in range(len(self.stops))] for i in range(len(self.stops))]

    def ballast(self):
        return max(0, self.initial_load - sum((t['weight'] for t in self.st.my_cargo())))

    def add(self, stop):
        self.stops.append(stop)
        return len(self.stops) - 1

    def add_events(self):
        st, cfg, graph = (self.st, self.cfg, self.graph)
        choices = []
        for inc in st.active_incidents():
            workers = max(inc['workers'], st.helpers(inc['id']))
            if inc['min_people'] > workers + 1:
                continue
            remaining = math.ceil(max(0, inc['need'] - inc['progress']))
            work = math.ceil(remaining / max(1, inc['workers'] + 1))
            distance = graph.path(st.here, inc['node'])[1]
            if not math.isfinite(distance) or work < 1:
                continue
            value = inc['reward'] / (workers + 1)
            if inc['type'] == 'generator' and st.escape['ready_at'] < 0:
                value += 700 / (1 + st.helpers(inc['id']))
            elif inc['type'] == 'gas' and st.escape['fuel'] < st.escape['fuel_required']:
                value += 450
            elif inc['type'] in ('rubble', 'chest') and st.escape['ready_at'] < 0:
                value += 120
            choices.append(Stop(inc['node'], 'event', dict(action='fix', id=inc['id'], work=remaining), service=work, latest=max(0, math.floor(inc['deadline'] - st.time)), value=value))
        for player in st.players:
            if player['id'] != st.you['id'] and player['state'] == 'downed':
                choices.append(Stop(player['node'], 'event', dict(action='revive', id=player['id']), service=3, latest=max(0, int(player['downed_timer']) - 2), value=250 * cfg.get('rescue', 1)))
        if st.you.get('ammo', 0) == 0:
            for node in st.nodes.values():
                if node['type'] == 'armory' and node.get('weapons', 0) > 0:
                    choices.append(Stop(node['id'], 'event', dict(action='weapon'), delta=int(st.rules['weapon_weight']), value=450 * cfg.get('weapon', 1)))
        if st.you.get('medkits', 0) < 1:
            for node in st.nodes.values():
                if node['type'] == 'supply' and node.get('medkits', 0) > 0:
                    choices.append(Stop(node['id'], 'event', dict(action='medkit'), delta=1, value=130))
        choices = [s for s in choices if min(s.latest, self.horizon) > s.service and math.isfinite(graph.path(st.here, s.node)[1])]
        choices.sort(key=lambda s: s.value / (graph.path(st.here, s.node)[1] / max(0.5, graph.speed(self.initial_load)) + s.service + 2), reverse=True)
        used_actions = set()
        for stop in choices:
            action = stop.ref['action']
            if action in ('weapon', 'medkit') and action in used_actions:
                continue
            used_actions.add(action)
            self.add(stop)
            if len(used_actions) >= 3 or sum((s.kind == 'event' for s in self.stops)) >= 3:
                break

    def reward_terms(self, task):
        value = float(task['reward'])
        if task['type'] == 'part':
            deficit = max(0, min(1, self.st.time / 420) - self.st.escape['parts'] / max(1, self.st.escape['parts_required']))
            value += (self.cfg['main'] + 100 * deficit) * task['weight']
        elif task['type'] == 'delivery':
            value *= self.cfg['process']
        elif task['type'] == 'fuel':
            value += 150
        elif task['type'] == 'rescue':
            value *= self.cfg.get('rescue', 1)
        if task['id'] in self.st.claimed_tasks() and task.get('carrier') != self.st.you['id']:
            value *= 0.35
        if task['id'] in self.own_targets:
            value *= 1.35
        value *= 0.96 + 0.08 * ((task['id'] * 73 + self.st.you['id'] * 131) % 97) / 97
        base = max(1, round(value))
        if task.get('due_at', -1) < 0:
            return (base, base, 0, 10000)
        multiplier = base / max(1, task['reward'])
        floor = min(base, max(0, round(task.get('min_reward', 0) * multiplier)))
        rate = max(0, round(task.get('late_per_sec', 0) * multiplier * self.cfg.get('late', 1)))
        if rate == 0:
            floor = base
        due = math.floor(task['due_at'] - self.st.time)
        return (base, floor, rate, due)

    def windows(self, index):
        return [(max(0, math.floor(a - self.st.time) - 1), min(self.horizon, math.ceil(b - self.st.time) + 1)) for a, b in self.st.danger_windows(self.stops[index].node) if b > self.st.time and a - self.st.time < self.horizon]

    def hint_values(self, order):
        clock, previous, carried = (0, 0, self.initial_load)
        times, loads, finish = ({0: 0}, {0: carried}, {})
        selected = set(order)
        home_seen = False
        for i in list(order) + [1]:
            stop = self.stops[i]
            if home_seen and stop.node != self.st.extraction:
                raise ValueError('split shelter visit')
            if not math.isfinite(self.distance[previous][i]):
                raise ValueError('unreachable')
            clock += self.stops[previous].service + math.ceil(math.ceil(self.distance[previous][i]) * math.ceil(1000 / self.graph.speed(carried)) / 1000)
            clock = max(clock, stop.release)
            for parent, child, delay in self.after:
                if child == i:
                    if parent not in finish:
                        raise ValueError('processing precedence')
                    clock = max(clock, finish[parent] + delay)
            if clock + stop.service > min(self.horizon, stop.latest):
                raise ValueError('time')
            if self.cfg.get('safe') and any((a < clock + stop.service and b > clock for a, b in self.windows(i))):
                raise ValueError('danger window')
            carried += stop.delta
            if not 0 <= carried <= self.capacity:
                raise ValueError('capacity')
            times[i], loads[i], finish[i] = (clock, carried, clock + stop.service)
            if stop.kind == 'drop' and stop.node == self.st.extraction:
                home_seen = True
            previous = i
        if not set(self.must) <= selected:
            raise ValueError('mandatory cargo')
        return (selected, times, loads)

    def seed_order(self):
        mandatory = sorted(self.must, key=lambda i: self.stops[i].node == self.st.extraction)
        candidates = [mandatory]
        if not mandatory:
            candidates += [[i] for i in range(2, len(self.stops)) if self.stops[i].kind == 'event']
            for pk, dr in self.pairs:
                order = [pk, dr]
                for parent, child, delay in self.after:
                    if parent == dr:
                        order += [child, self.pick_to_drop[child]]
                candidates.append(order)
            for group, _ in self.sets:
                if all((self.stops[i].kind == 'pick' and self.stops[self.pick_to_drop[i]].node == self.st.extraction for i in group)):
                    candidates.append(list(group) + [self.pick_to_drop[i] for i in group])
        feasible = []
        for order in candidates:
            try:
                self.hint_values(order)
            except ValueError:
                continue
            steps, value = self.to_plan(order)
            if steps or not order:
                feasible.append((value.get('utility', -100000000.0) if order else 0, order))
        return max(feasible, default=(0, []), key=lambda item: item[0])[1]

    def to_plan(self, order):
        st, graph = (self.st, self.graph)
        clock, at, load = (st.time, st.here, self.initial_load)
        steps, finished, picked, score, danger, utility = ([], {}, [], 0.0, 0.0, 0.0)
        for index in order:
            stop = self.stops[index]
            route, arrival, risk = graph.route(at, stop.node, clock, load)
            if not math.isfinite(arrival):
                return ([], dict(reason='unreachable'))
            steps.extend(route)
            clock = arrival
            release = st.time + stop.release
            if stop.kind == 'pick' and stop.ref.get('status') == 'locked':
                parent = stop.ref.get('after')
                if parent not in finished:
                    return ([], dict(reason='missing_processing_parent'))
                release = max(release, finished[parent] + stop.ref.get('proc_delay', 0) + 0.25)
            if release > clock:
                steps.append(wait(release - clock + 0.25))
                clock = release + 0.25
            if self.cfg.get('safe'):
                for a, b in st.danger_windows(stop.node):
                    if a < clock + stop.service and b > clock and (b - clock < 12):
                        steps.append(wait(b - clock + 1))
                        clock = b + 1
            if clock + stop.service > st.time + stop.latest + 0.5:
                return ([], dict(reason='expired_plan'))
            if not 0 <= load + stop.delta <= self.capacity:
                return ([], dict(reason='capacity'))
            load += stop.delta
            if stop.kind == 'pick':
                steps.append(pickup(stop.ref['id']))
                picked.append(stop.ref['id'])
            elif stop.kind == 'drop':
                steps.append(dropoff(stop.ref['id']))
                finished[stop.ref['id']] = clock + stop.service
                score += st.value_of(stop.ref, clock + stop.service)
                base, floor, rate, due = self.reward_terms(stop.ref)
                utility += max(floor, base - rate * max(0, clock + stop.service - st.time - due))
            else:
                action = stop.ref['action']
                steps.append(fix(stop.ref['id'], stop.ref['work'] + 0.2) if action == 'fix' else revive(stop.ref['id']) if action == 'revive' else take_weapon() if action == 'weapon' else take_medkit())
                utility += stop.value
            clock += stop.service
            danger += risk
            at = stop.node
        route, arrival, risk = graph.route(at, st.extraction, clock, load)
        steps.extend(route)
        if not math.isfinite(arrival) or arrival > st.raw['duration'] - 4:
            return ([], dict(reason='cannot_return_before_end'))
        for group, bonus in self.sets:
            if all((self.stops[i].ref['id'] in finished for i in group)):
                utility += bonus * self.cfg['sets']
        utility -= (arrival - st.time) * self.cfg['time'] + (danger + risk) * self.cfg['risk']
        if len(steps) > st.rules['max_plan_steps']:
            return ([], dict(reason='step_limit'))
        return (steps, dict(reason='optimized', picks=picked, expected_seconds=round(arrival - st.time, 2), predicted_base_score=round(score), route_risk=round(danger + risk, 2), utility=round(utility, 2)))

def make_policy():
    return Policy('flow_controller')
