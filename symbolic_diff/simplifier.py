from .parser import ASTNode, sexp_to_ast, ast_to_sexp
import math


def prettify_float(n: float) -> str:
    """Format a number as a string, using its integer representation when possible."""
    if n.is_integer():
        return str(int(n))
    return str(n)


def simplify(sexp: str) -> str:
    """
    Simplify a LISP s-expression by converting to AST, simplifying, and converting back.
    Useful mainly for testing the simplifier.

    Example:
        >>> simplify("(/ (* (* x 2) 2) 4)")
        "x"
    """
    ast = sexp_to_ast(sexp)
    simplified = simplify_ast(ast)
    return ast_to_sexp(simplified)


def simplify_ast(ast: ASTNode) -> ASTNode:
    """Internal function to simplify an AST."""
    # Atomic (base) case
    if ast.type in ("NUMBER", "VARIABLE"):
        return ast

    # Because we must handle ALL errors
    if ast.type != "OPERATOR":
        raise ValueError(f"Unknown node type: {ast.type}")

    # Recursively simplify children first
    simplified_children = [simplify_ast(child) for child in ast.children]

    if ast.value == "+":
        # Remove 0 terms
        non_zero_terms = [
            child
            for child in simplified_children
            if not (child.type == "NUMBER" and float(child.value) == 0)
        ]

        if len(non_zero_terms) == 0:
            return ASTNode("NUMBER", "0")

        if len(non_zero_terms) == 1:
            return non_zero_terms[0]

        # Combine number terms
        number_sum = 0
        terms = []
        for child in non_zero_terms:
            if child.type == "NUMBER":
                number_sum += float(child.value)
            else:
                terms.append(child)

        if number_sum != 0:
            terms.append(ASTNode("NUMBER", prettify_float(number_sum)))

        if len(terms) == 0:
            return ASTNode("NUMBER", "0")

        if len(terms) == 1:
            return terms[0]

        return ASTNode("OPERATOR", "+", terms)

    elif ast.value == "*":
        # Check for zeros
        if any(
            child.type == "NUMBER" and float(child.value) == 0
            for child in simplified_children
        ):
            return ASTNode("NUMBER", "0")

        # Remove 1 terms
        non_one_terms = [
            child
            for child in simplified_children
            if not (child.type == "NUMBER" and float(child.value) == 1)
        ]

        if len(non_one_terms) == 0:
            return ASTNode("NUMBER", "1")

        if len(non_one_terms) == 1:
            return non_one_terms[0]

        # Combine number terms
        number_product = 1
        terms = []
        for child in non_one_terms:
            if child.type == "NUMBER":
                number_product *= float(child.value)
            else:
                terms.append(child)

        if number_product != 1:
            terms.append(ASTNode("NUMBER", prettify_float(number_product)))

        if len(terms) == 1:
            return terms[0]

        return ASTNode("OPERATOR", "*", terms)

    elif ast.value == "^":
        if len(simplified_children) != 2:
            raise ValueError("Power operator takes exactly 2 arguments")

        base, exponent = simplified_children

        # Anything ^ 0 = 1
        if exponent.type == "NUMBER" and float(exponent.value) == 0:
            return ASTNode("NUMBER", "1")

        # Anything ^ 1 = itself
        if exponent.type == "NUMBER" and float(exponent.value) == 1:
            return base

        # 0 ^ anything = 0 (except 0 and negative exponents)
        if (
            base.type == "NUMBER"
            and float(base.value) == 0
            and exponent.type == "NUMBER"
            and float(exponent.value) > 0
        ):
            return ASTNode("NUMBER", "0")

        # 1 ^ anything = 1
        if base.type == "NUMBER" and float(base.value) == 1:
            return ASTNode("NUMBER", "1")

        # If both are numbers, compute the result
        if base.type == "NUMBER" and exponent.type == "NUMBER":
            try:
                result = float(base.value) ** float(exponent.value)
                if math.isinf(result) or math.isnan(result):
                    # If result is too large/undefined, leave as is
                    return ASTNode("OPERATOR", "^", [base, exponent])
                return ASTNode("NUMBER", prettify_float(result))
            except OverflowError:
                # If computation overflows, leave expression as is
                return ASTNode("OPERATOR", "^", [base, exponent])

        return ASTNode("OPERATOR", "^", [base, exponent])

    elif ast.value == "/":
        if len(simplified_children) != 2:
            raise ValueError("Division operator takes exactly 2 arguments")

        numerator, denominator = simplified_children

        # x/x = 1
        # Use ASTNode's __eq__ to catch both atomic numbers & variables and
        # expressions like (x + y + 1) / (x + y + 1)
        if numerator == denominator:
            return ASTNode("NUMBER", "1")

        # x/1 = x
        if denominator.type == "NUMBER" and float(denominator.value) == 1:
            return numerator

        # 0/x = 0 (assuming x != 0)
        if numerator.type == "NUMBER" and float(numerator.value) == 0:
            return ASTNode("NUMBER", "0")

        # If both operands are numbers, compute the division
        if numerator.type == "NUMBER" and denominator.type == "NUMBER":
            if float(denominator.value) == 0:
                raise ValueError("Division by zero")
            result = float(numerator.value) / float(denominator.value)
            return ASTNode("NUMBER", prettify_float(result))

        return ASTNode("OPERATOR", "/", [numerator, denominator])

    elif ast.value == "-":
        # Convert subtraction to addition with negative number
        if len(simplified_children) != 2:
            raise ValueError("Subtraction operator takes exactly 2 arguments")

        left_op, right_op = simplified_children

        if right_op.type == "NUMBER":
            neg_right_op = ASTNode("NUMBER", str(-float(right_op.value)))
        else:
            neg_right_op = ASTNode("OPERATOR", "*", [ASTNode("NUMBER", "-1"), right_op])

        return simplify_ast(ASTNode("OPERATOR", "+", [left_op, neg_right_op]))

    # If no simplification rules apply, return node with simplified children
    return ASTNode(ast.type, ast.value, simplified_children)
