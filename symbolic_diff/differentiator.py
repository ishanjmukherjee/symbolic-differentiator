from .parser import parse, ASTNode
from .tokenizer import tokenize
from .simplifier import simplify_ast


def ast_to_sexp(ast: ASTNode) -> str:
    """Convert AST back to s-expression string."""
    if ast.type == "NUMBER":
        return ast.value
    elif ast.type == "VARIABLE":
        return ast.value
    elif ast.type == "OPERATOR":
        children = " ".join(ast_to_sexp(child) for child in ast.children)
        return f"({ast.value} {children})"
    raise ValueError(f"Unknown node type: {ast.type}")


def sexp_to_ast(sexp: str) -> ASTNode:
    """Convert s-expression string to AST."""
    return parse(tokenize(sexp))


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

        raise ValueError(f"Operator not implemented: {ast.value}")

    raise ValueError(f"Unknown node type: {ast.type}")