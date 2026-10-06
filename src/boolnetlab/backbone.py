import ast

from pysat.solvers import Glucose3


class TseitinEncoder:
    """Encodes a Python-syntax boolean expression (&, |, ~) into CNF via Tseitin."""

    def __init__(self):
        self.var_ids = {}   # variable name -> CNF var id
        self.next_id = 1
        self.clauses = []

    def _new_var(self):
        vid = self.next_id
        self.next_id += 1
        return vid

    def var_id(self, name):
        if name not in self.var_ids:
            self.var_ids[name] = self._new_var()
        return self.var_ids[name]

    def encode(self, node):
        """Return a literal (int) representing the truth value of `node`."""
        if isinstance(node, ast.Expression):
            return self.encode(node.body)
        if isinstance(node, ast.Name):
            return self.var_id(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Invert):
            return -self.encode(node.operand)
        if isinstance(node, ast.BinOp):
            a = self.encode(node.left)
            b = self.encode(node.right)
            g = self._new_var()
            if isinstance(node.op, ast.BitAnd):
                self.clauses += [[-g, a], [-g, b], [g, -a, -b]]
            elif isinstance(node.op, ast.BitOr):
                self.clauses += [[g, -a], [g, -b], [-g, a, b]]
            elif isinstance(node.op, ast.BitXor):
                self.clauses += [[-g, a, b], [-g, -a, -b], [g, -a, b], [g, a, -b]]
            else:
                raise ValueError(f"Unsupported operator: {ast.dump(node.op)}")
            return g
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            g = self._new_var()
            self.clauses.append([g] if node.value else [-g])
            return g
        raise ValueError(f"Unsupported expression node: {ast.dump(node)}")


def _parse(expr):
    tree = ast.parse(expr, mode="eval")
    enc = TseitinEncoder()
    root_lit = enc.encode(tree)
    enc.clauses.append([root_lit])  # constrain the formula to be True
    return enc


def backbone(expr):
    """
    Return {var_name: fixed_bool} for every variable that has the same value
    in every satisfying assignment of `expr`. Returns None if `expr` is
    unsatisfiable (never evaluates to True).
    """
    enc = _parse(expr)
    with Glucose3(bootstrap_with=enc.clauses) as solver:
        if not solver.solve():
            return None

        model_val = {abs(l): (l > 0) for l in solver.get_model()}
        candidates = set(enc.var_ids.values())
        fixed = {}

        while candidates:
            v = candidates.pop()
            assumption = -v if model_val[v] else v  # force the opposite value
            if solver.solve(assumptions=[assumption]):
                new_val = {abs(l): (l > 0) for l in solver.get_model()}
                # any remaining candidate whose value differs is proven non-fixed
                for v2 in list(candidates):
                    if new_val[v2] != model_val[v2]:
                        candidates.discard(v2)
            else:
                fixed[v] = model_val[v]

        id_to_name = {vid: name for name, vid in enc.var_ids.items()}
        return {id_to_name[v]: val for v, val in fixed.items()}


if __name__ == "__main__":
    print(backbone("((X1 & X2) | (~X1 & X3)) & X3"))
