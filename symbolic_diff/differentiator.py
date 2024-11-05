from .parser import ASTNode, ast_to_sexp, sexp_to_ast
from .simplifier import simplify_ast


def validate_variable(var: str) -> None:
    """
    Validate variable name and raise descriptive errors if invalid.

    Raises:
        ValueError with specific error message describing the issue
    """
    # We check that length of var is zero, not `not var`, because we know that
    # implicit falsiness is fraught with peril
    if len(var) == 0:
        raise ValueError("Variable name cannot be empty")

    if any(c.isspace() for c in var):
        raise ValueError(f"Variable name contains whitespace: {var}")

    if not (var[0].isalpha() or var[0] == "_"):
        raise ValueError(
            f"Variable name must start with letter or underscore, got: {var}"
        )

    invalid_chars = [c for c in var if not (c.isalnum() or c == "_")]
    if len(invalid_chars) > 0:
        raise ValueError(
            f"Variable name '{var}' contains invalid characters: {invalid_chars}"
        )


def differentiate(expression: str, var: str = "x") -> str:
    """
    Differentiate a LISP s-expression with respect to a variable.

    Args:
        expression (str): A LISP s-expression like "(+ x 6)"
        var (str): Variable to differentiate with respect to

    Returns:
        str: The derivative as a LISP s-expression
    """
    validate_variable(var)
    ast = sexp_to_ast(expression)
    # Remove whitespace from variable, so that differentiate("x", "x\n") works
    # as intended (gives "1")
    result = differentiate_ast(ast, var)
    simplified_result = simplify_ast(result)
    return ast_to_sexp(simplified_result)


def differentiate_ast(ast: ASTNode, var: str) -> ASTNode:
    """Internal function to differentiate an AST."""
    if ast.type == "NUMBER":
        return ASTNode("NUMBER", "0")

    if ast.type == "VARIABLE":
        return ASTNode("NUMBER", "1" if ast.value == var else "0")

    if ast.type == "OPERATOR":
        if ast.value == "+":
            return ASTNode(
                "OPERATOR",
                "+",
                [differentiate_ast(child, var) for child in ast.children],
            )

        elif ast.value == "*":
            # For (* u v w ...), d/dx(u*v*w) = (u'*v*w) + (u*v'*w) + (u*v*w') +
            # ...
            terms = []
            for i in range(len(ast.children)):
                # Only the i-th factor is differentiated
                factors = []
                for j, child in enumerate(ast.children):
                    if j == i:
                        factors.append(differentiate_ast(child, var))
                    else:
                        factors.append(child)
                terms.append(ASTNode("OPERATOR", "*", factors))

            if len(terms) == 1:
                return terms[0]
            return ASTNode("OPERATOR", "+", terms)

        elif ast.value == "^":
            if len(ast.children) != 2:
                raise ValueError("Power operator takes exactly 2 arguments")

            base, exponent = ast.children

            # Constant exponents
            if exponent.type == "NUMBER":
                n = int(exponent.value)
                if n == 0:
                    return ASTNode("NUMBER", "0")

                # Get derivative of base
                du_dx = differentiate_ast(base, var)
                if n == 1:
                    return du_dx

                # d/dx(u^n) = n*u^(n-1)*u'
                return ASTNode(
                    "OPERATOR",
                    "*",
                    [
                        ASTNode("NUMBER", str(n)),
                        ASTNode("OPERATOR", "^", [base, ASTNode("NUMBER", str(n - 1))]),
                        du_dx,
                    ],
                )

            # TODO: Variable exponents (requires natural log)
            # d/dx(u^v) = u^v * (v'*ln(u) + v*u'/u)
            raise ValueError("Non-constant exponents not yet supported")

        elif ast.value == "/":
            # d/dx(u/v) = (u'v - uv')/(v^2)
            if len(ast.children) != 2:
                raise ValueError("Division operator takes exactly 2 arguments")

            u, v = ast.children
            if v.type == "NUMBER" and v.value == "0":
                raise ValueError("Cannot divide by zero")

            du_dx = differentiate_ast(u, var)
            dv_dx = differentiate_ast(v, var)

            return ASTNode(
                "OPERATOR",
                "/",
                [
                    # Numerator: u'v - uv'
                    ASTNode(
                        "OPERATOR",
                        "-",
                        [
                            ASTNode("OPERATOR", "*", [du_dx, v]),
                            ASTNode("OPERATOR", "*", [u, dv_dx]),
                        ],
                    ),
                    # Denominator: v^2
                    ASTNode("OPERATOR", "^", [v, ASTNode("NUMBER", "2")]),
                ],
            )

        raise ValueError(f"Operator not implemented: {ast.value}")

    raise ValueError(f"Unknown node type: {ast.type}")
