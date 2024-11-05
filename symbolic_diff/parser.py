from .tokenizer import tokenize


class ASTNode:
    # We use type_ with the underscore to avoid shadowing Python's built-in
    # type()
    def __init__(self, type_: str, value: str, children=None):
        self.type = type_
        self.value = value
        # A refresher on why the below works:
        # If children is None, it evaluates as falsy. So, or moves on to
        # evaluate the next value. So it's [], the second value, that's
        # returned.
        # Boolean operators can be seen as control flow!
        self.children = children or []

    # Useful primarily for testing
    def __eq__(self, other):
        if not isinstance(other, ASTNode):
            return False
        return (
            self.type == other.type
            and self.value == other.value
            and self.children == other.children
        )

    # Useful primarily for debugging
    def __repr__(self):
        # By convention, a string that could be eval()ed to recreate the object
        return f"ASTNode({self.type}, {self.value}, {self.children})"


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


def parse(tokens):
    """Convert a sequence of tokens into an AST.

    Args:
        tokens (list[Token]): List of tokens from the tokenizer

    Returns:
        ASTNode: The root node of the AST

    Example:
        >>> tokens = tokenize("(+ x 6)")
        >>> parse(tokens)
        ASTNode('OPERATOR', '+', [
            ASTNode('VARIABLE', 'x', []),
            ASTNode('NUMBER', '6', [])
        ])
    """

    # Recursive helper function that does the actual parsing
    def parse_expression(tokens, pos):
        if pos >= len(tokens):
            raise ValueError("Unexpected end of input")

        token = tokens[pos]

        if token.type == "NUMBER":
            return ASTNode("NUMBER", token.value), pos + 1
        elif token.type == "VARIABLE":
            return ASTNode("VARIABLE", token.value), pos + 1
        elif token.type == "LPAREN":
            if pos + 1 >= len(tokens):
                raise ValueError("Unexpected end of input after (")

            operator = tokens[pos + 1]
            if operator.type != "OPERATOR":
                raise ValueError(f"Expected operator after (, got {operator.type}")

            node = ASTNode("OPERATOR", operator.value)
            current_pos = pos + 2

            while current_pos < len(tokens) and tokens[current_pos].type != "RPAREN":
                child, new_pos = parse_expression(tokens, current_pos)
                node.children.append(child)
                current_pos = new_pos

            if current_pos >= len(tokens) or tokens[current_pos].type != "RPAREN":
                raise ValueError("Missing closing parenthesis")

            return node, current_pos + 1

        raise ValueError(f"Unexpected token: {token}")

    ast, pos = parse_expression(tokens, 0)
    if pos != len(tokens):
        raise ValueError("Extra tokens after expression")
    return ast
