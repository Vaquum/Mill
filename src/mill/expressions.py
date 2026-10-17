"""Factor native Polars producer plans without evaluating their intermediates.

Only the frame operations below are admitted. Opaque Python kernels and other
frame operations stay on the producer's ordinary eager path.
"""
from __future__ import annotations

import io
import json

import polars as pl


class NotPlannable(Exception):
    pass


class Graph:
    def __init__(self, raw):
        self.raw = raw
        self.nodes = {}
        self.interned = {}
        self.prefix = "_mill_shared_"
        while any(c.startswith(self.prefix) for c in raw.columns):
            self.prefix += "_"

    def frame(self):
        return Frame(self, {name: name for name in self.raw.columns})

    def expression(self, expr, columns):
        if not isinstance(expr, pl.Expr):
            raise NotPlannable()
        try:
            name = expr.meta.output_name()
            tree = json.loads(expr.meta.serialize(format="json"))
        except pl.exceptions.PolarsError as exc:
            raise NotPlannable() from exc
        if "Alias" in tree:
            tree = tree["Alias"][0]
        dependencies = set()

        def visit(node):
            if isinstance(node, dict):
                if "AnonymousFunction" in node or "Random" in node:
                    raise NotPlannable()
                if "Column" in node:
                    original = node["Column"]
                    if original not in columns:
                        raise NotPlannable()
                    shared = columns[original]
                    if shared in self.nodes:
                        dependencies.add(shared)
                    return {"Column": shared}
                return {k: visit(v) for k, v in node.items()}
            if isinstance(node, list):
                return [visit(v) for v in node]
            return node

        tree = visit(tree)
        key = json.dumps(tree, sort_keys=True)
        shared = self.interned.get(key)
        if shared is None:
            restored = pl.Expr.deserialize(io.StringIO(json.dumps(tree)), format="json")
            # Polars' JSON float parser need not round every decimal literal
            # to the original bit pattern. Reject such plans, rather than
            # changing an indicator by even one ULP.
            if json.loads(restored.meta.serialize(format="json")) != tree:
                raise NotPlannable()
            shared = f"{self.prefix}{len(self.nodes)}"
            self.interned[key] = shared
            self.nodes[shared] = (restored, dependencies)
        return name, shared

    def collect(self, parts):
        needed = set()

        def require(name):
            if name in self.nodes and name not in needed:
                needed.add(name)
                for parent in self.nodes[name][1]:
                    require(parent)

        for part in parts:
            for node in part.mapping.values():
                require(node)
        frame, done = self.raw.lazy(), set()
        while needed - done:
            ready = [name for name in self.nodes if name in needed - done
                     and self.nodes[name][1] <= done]
            expressions = [self.nodes[name][0].alias(name) for name in ready]
            frame = frame.with_columns(expressions)
            done.update(ready)
        outputs, groups = [], []
        for part in parts:
            group = []
            for name, node in part.mapping.items():
                temporary = f"{self.prefix}output_{len(outputs)}"
                outputs.append(pl.col(node).alias(temporary))
                group.append((temporary, name))
            groups.append(group)
        combined = frame.select(outputs).collect()
        return [combined.select(pl.col(temporary).alias(name) for temporary, name in group)
                for group in groups]


class Frame:
    """A deliberately small, expression-only frame protocol for registry producers."""
    def __init__(self, graph, mapping):
        self.graph, self.mapping = graph, mapping

    @property
    def columns(self):
        return list(self.mapping)

    def __getattr__(self, name):
        raise NotPlannable(name)

    def with_columns(self, *exprs, **named):
        if named:
            raise NotPlannable()
        if len(exprs) == 1 and isinstance(exprs[0], (list, tuple)):
            exprs = exprs[0]
        pairs = [self.graph.expression(expr, self.mapping) for expr in exprs]
        added = dict(pairs)
        if len(added) != len(pairs):
            raise NotPlannable()
        return Frame(self.graph, self.mapping | added)

    def drop(self, *names, **kwargs):
        if kwargs:
            raise NotPlannable()
        if len(names) == 1 and isinstance(names[0], (list, tuple)):
            names = names[0]
        if any(name not in self.mapping for name in names):
            raise NotPlannable()
        return Frame(self.graph, {k: v for k, v in self.mapping.items() if k not in names})

    def select(self, *names):
        if len(names) == 1 and isinstance(names[0], (list, tuple)):
            names = names[0]
        selected = {}
        for name in names:
            if isinstance(name, pl.Expr) and name.meta.is_column():
                name = name.meta.output_name()
            if not isinstance(name, str) or name not in self.mapping or name in selected:
                raise NotPlannable()
            selected[name] = self.mapping[name]
        return Frame(self.graph, selected)

    def rename(self, mapping):
        if not isinstance(mapping, dict):
            raise NotPlannable()
        renamed = {mapping.get(k, k): v for k, v in self.mapping.items()}
        if len(renamed) != len(self.mapping):
            raise NotPlannable()
        return Frame(self.graph, renamed)


def evaluate(raw, requests, *, errors=False):
    """Evaluate a batch, sharing identical native expression nodes across producers."""
    graph, planned, positions, results = Graph(raw), [], [], [None] * len(requests)
    for i, (producer, params) in enumerate(requests):
        if getattr(producer, "_mill_native_plan", False):
            try:
                part = producer(graph.frame(), **params)
            except (NotPlannable, TypeError, ValueError, AttributeError, pl.exceptions.PolarsError):
                pass
            else:
                if isinstance(part, Frame):
                    planned.append(part)
                    positions.append(i)
                    continue
        try:
            results[i] = producer(raw, **params)
        except Exception as exc:
            if not errors:
                raise
            results[i] = exc
    if planned:
        try:
            computed = graph.collect(planned)
        except pl.exceptions.PolarsError:
            # Preserve per-factory failures in catalogs: one invalid planned
            # expression must not suppress other factories in the batch.
            computed = []
            for i in positions:
                producer, params = requests[i]
                try:
                    computed.append(producer(raw, **params))
                except Exception as exc:
                    if not errors:
                        raise
                    computed.append(exc)
        for i, part in zip(positions, computed):
            results[i] = part
    return results
