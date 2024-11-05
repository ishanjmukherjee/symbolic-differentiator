import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from symbolic_diff.differentiator import differentiate
from symbolic_diff.parser import ASTNode, ast_to_sexp, sexp_to_ast
from symbolic_diff.simplifier import simplify


def exprs_equivalent(expr1: str, expr2: str, test_values=None, var: str = "x") -> bool:
    """
    Test if two expressions are mathematically equivalent by evaluating them
    with test values for variables.

    This is a rough-and-dirty way of checking mathematical equivalence. The
    greater the number of test values, the more confident we can be that the
    function being tested works.
    """
    if test_values is None:
        test_values = [0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 314159, -314159]

    def eval_expr(expr: str, var_value: float) -> float:
        ast = sexp_to_ast(expr)

        def evaluate_ast(node: ASTNode) -> float:
            if node.type == "NUMBER":
                return float(node.value)
            elif node.type == "VARIABLE":
                return var_value if node.value == var else 0.0
            elif node.type == "OPERATOR":
                children = [evaluate_ast(child) for child in node.children]
                if node.value == "+":
                    return sum(children)
                if node.value == "*":
                    result = 1.0
                    for child in children:
                        result *= float(child)
                    return result
                if node.value == "^":
                    return children[0] ** children[1]
                if node.value == "/":
                    return children[0] / children[1]
                if node.value == "-":
                    return children[0] - children[1]
                raise ValueError(f"Unknown operator: {node.value}")
            raise ValueError(f"Unknown node type: {node.type}")

        try:
            return evaluate_ast(ast)
        except (ZeroDivisionError, OverflowError, ValueError):
            return float("nan")

    return all(
        math.isnan(eval_expr(expr1, x))
        and math.isnan(eval_expr(expr2, x))
        or abs(eval_expr(expr1, x) - eval_expr(expr2, x)) < 1e-10
        for x in test_values
    )


operators = st.sampled_from("+-*/^")
numbers = st.from_regex(r"^[+-]?(\d+\.?\d*|\.\d+)$").map(str.strip)
variables = st.from_regex(r"^[a-z_][a-z0-9_]*$").map(str.strip)


# Strategy for generating s-expressions
@st.composite
def sexprs(draw, max_depth=2):
    """Generate random s-expressions with bounded depth."""
    if max_depth == 0:
        # Base case: return either a number or variable
        return draw(st.one_of(numbers, variables))

    # Recursive case: generate an operator expression
    operator = draw(st.sampled_from(["+", "*"]))
    if operator == "^":
        # For power, always generate number as exponent to avoid unsupported cases
        base = draw(sexprs(max_depth=max_depth - 1))
        exponent = draw(st.integers(min_value=0, max_value=5))
        return f"(^ {base} {exponent})"
    else:
        num_operands = draw(st.integers(min_value=2, max_value=5))
        operands = [draw(sexprs(max_depth=max_depth - 1)) for _ in range(num_operands)]
        return f"({operator} {' '.join(operands)})"


@given(sexprs())
def test_ast_conversion_roundtrip(expr):
    """Test that converting to AST and back preserves the expression."""
    ast = sexp_to_ast(expr)
    result = ast_to_sexp(ast)
    # Normalize spaces and parentheses
    expected = expr.replace("(", " ( ").replace(")", " ) ")
    expected = " ".join(expected.split())
    result = result.replace("(", " ( ").replace(")", " ) ")
    result = " ".join(result.split())
    assert result == expected


@given(sexprs(), variables)
def test_derivative_structure(expr, var):
    """Test that the derivative maintains valid s-expression structure."""
    result = differentiate(expr, var=var)
    # Verify the result can be parsed back to AST
    try:
        ast = sexp_to_ast(result)
        # Verify it can be converted back to string
        _ = ast_to_sexp(ast)
    except Exception as e:
        pytest.fail(f"Invalid derivative structure: {result}\nError: {str(e)}")


@given(numbers)
def test_constant_derivative(num):
    """The derivative of any constant is 0."""
    result = differentiate(num)
    assert result == "0"


@given(variables)
def test_variable_derivative(var):
    """The derivative of a variable is 1 if it's the target, 0 otherwise."""
    # When differentiating with respect to the variable
    result = differentiate(var, var=var)
    assert result == "1"

    # When differentiating with respect to a different variable
    other_var = "x" if var != "x" else "y"
    result = differentiate(var, var=other_var)
    assert result == "0"


@given(st.lists(numbers, min_size=2, max_size=5))
def test_addition_multiple_constants(nums):
    """The derivative of a sum of constants is 0."""
    expr = f"(+ {' '.join(nums)})"
    result = differentiate(expr)
    expected_terms = ["0" for n in nums]
    # Unsimplified and simplified forms
    expected = [f"(+ {' '.join(expected_terms)})", "0"]
    assert result in expected


@given(st.lists(variables, min_size=2, max_size=5))
def test_addition_multiple_variables(vars):
    """Test derivative of sum of multiple variables."""
    expr = f"(+ {' '.join(vars)})"
    target_var = vars[0]  # Use first variable as target
    result = differentiate(expr, var=target_var)

    # Non-simplified result: "1" if variables match (e.g., differentiating "x"
    # w.r.t. "x") else "0" (e.g., differentiating "y" w.r.t. "x")
    # Simplified result: just the sum
    expected_terms = [("1" if v == target_var else "0") for v in vars]
    expected = [
        f"(+ {' '.join(expected_terms)})",
        f"{sum(int(expected_term) for expected_term in expected_terms)}",
    ]

    assert result in expected


@given(sexprs(max_depth=1), sexprs(max_depth=1), variables)
def test_sum_rule_property(expr1, expr2, var):
    """Property-based test for d/dx(f + g) = d/dx(f) + d/dx(g)"""
    sum_expr = f"(+ {expr1} {expr2})"
    sum_derivative = differentiate(sum_expr, var=var)

    part1 = differentiate(expr1, var=var)
    part2 = differentiate(expr2, var=var)

    assert exprs_equivalent(sum_derivative, f"(+ {part1} {part2})", var)


@given(st.lists(numbers, min_size=2, max_size=5))
def test_product_constants(nums):
    """Test that the derivative of a product of constants is 0."""
    expr = f"(* {' '.join(nums)})"
    result = differentiate(expr)
    # Should be 0
    root = sexp_to_ast(result)
    assert root.value == "0" and root.type == "NUMBER"


@given(variables, numbers)
def test_product_rule_basic(var, num):
    """Test basic product rule with variable and constant."""
    expr = f"(* {var} {num})"
    # result and expected should be the same, regardless of how simplify is
    # implemented. This is because internally, differentiate uses the same
    # simplification technique.
    result = differentiate(expr, var=var)
    expected = simplify(f"(+ (* 1 {num}) (* {var} 0))")
    assert sexp_to_ast(result) == sexp_to_ast(expected)


@given(sexprs(max_depth=2), sexprs(max_depth=2), variables)
def test_product_rule_property(expr1, expr2, var):
    """Property-based test for product rule: d/dx(f*g) = f'g + fg'"""
    product_expr = f"(* {expr1} {expr2})"
    product_derivative = differentiate(product_expr, var=var)

    # Compute parts separately
    d_expr1 = differentiate(expr1, var=var)
    d_expr2 = differentiate(expr2, var=var)

    # Build expected result: (+ (* f' g) (* f g'))
    expected = simplify(f"(+ (* {d_expr1} {expr2}) (* {expr1} {d_expr2}))")

    assert sexp_to_ast(product_derivative) == sexp_to_ast(expected)


def test_power_rule_basic():
    """Test basic power rule cases."""
    # x^0 -> 0
    assert differentiate("(^ x 0)") == "0"

    # x^1 -> 1
    assert differentiate("(^ x 1)") == "1"

    # x^2 -> 2x^1
    result = differentiate("(^ x 2)")
    expected = simplify("(* 2 (^ x 1) 1)")
    assert result == expected
    assert sexp_to_ast(result) == sexp_to_ast(expected)

    # x^3 -> 3x^2
    result = differentiate("(^ x 3)")
    expected = simplify("(* 3 (^ x 2) 1)")
    assert result == expected
    assert sexp_to_ast(result) == sexp_to_ast(expected)


@given(st.integers(min_value=0, max_value=10))
def test_power_rule_variable(n):
    """Test power rule with variable base and constant integer exponent."""
    expr = f"(^ x {n})"
    result = differentiate(expr)

    if n == 0:
        assert result == "0"
    elif n == 1:
        assert result == "1"
    else:
        # Should be n * x^(n-1)
        expected = simplify(f"(* {n} (^ x {n-1}))")
        assert result == expected
        assert sexp_to_ast(result) == sexp_to_ast(expected)


@given(variables, st.integers(min_value=0, max_value=5))
def test_power_rule_any_variable(var, n):
    """Test power rule works with any variable."""
    expr = f"(^ {var} {n})"
    result = differentiate(expr, var=var)

    if n == 0:
        assert result == "0"
    elif n == 1:
        assert result == "1"
    else:
        expected = simplify(f"(* {n} (^ {var} {n-1}) 1)")
        assert result == expected
        assert sexp_to_ast(result) == sexp_to_ast(expected)


@given(sexprs(max_depth=2), st.integers(min_value=0, max_value=5), variables)
def test_power_rule_compound_base(base_expr, n, var):
    """Test power rule with compound bases produces parsable sexprs."""
    try:
        expr = f"(^ {base_expr} {n})"
        result = differentiate(expr, var=var)

        # Verify parsability
        ast = sexp_to_ast(result)

        if n == 0:
            assert result == "0"
        else:
            # Verify validity of AST structure
            _ = ast_to_sexp(ast)
    except Exception as e:
        pytest.fail(f"Failed with expr={expr}, n={n}: {str(e)}")


@given(st.integers(min_value=1, max_value=5), st.integers(min_value=2, max_value=5))
def test_power_rule_chain_rule(factor, exponent):
    """Test power rule combined with chain rule."""
    # d/dx((2x)^n)
    expr = f"(^ (* {factor} x) {exponent})"
    result = differentiate(expr)

    # n * (2x)^(n-1) * 2
    expected = simplify(
        f"(* {exponent} (^ (* {factor} x) {exponent-1}) (+ (* 0 x) (* {factor} 1)))"
    )
    assert result == expected
    assert sexp_to_ast(result) == sexp_to_ast(expected)


def test_power_invalid():
    """Test invalid power expressions."""
    # Non-constant exponents not yet implemented
    with pytest.raises(ValueError):
        differentiate("(^ x x)")
    with pytest.raises(ValueError):
        differentiate("(^ x y)")
    # Too few arguments
    with pytest.raises(ValueError):
        differentiate("(^ x)")
    # Too many arguments
    with pytest.raises(ValueError):
        differentiate("(^ x 2 3)")


@given(variables)
def test_division_by_one(var):
    """Test that differentiating x/1 gives 1."""
    expr = f"(/ {var} 1)"
    result = differentiate(expr, var)
    expected = "1"
    assert result == expected


@given(variables)
def test_division_zero_numerator(var):
    """Test that differentiating 0/x gives 0 for any variable x."""
    expr = f"(/ 0 {var})"
    result = differentiate(expr)
    assert result == "0"


@given(variables)
def test_division_same_terms(var):
    """Test that differentiating x/x gives 0 (quotient rule where f=g)."""
    expr = f"(/ {var} {var})"
    result = differentiate(expr, var=var)
    assert exprs_equivalent(
        result, "0", var=var, test_values=[1.0, 2.0, -1.0, -2.0, 0.5, -0.5]
    )


@given(sexprs(max_depth=2), sexprs(max_depth=2), variables)
def test_quotient_rule_property(expr1, expr2, var):
    """Property-based test for quotient rule: d/dx(f/g) = (f'g - fg')/(g^2)"""
    try:
        quotient_expr = f"(/ {expr1} {expr2})"
        quotient_derivative = differentiate(quotient_expr, var=var)

        # Compute parts separately
        d_expr1 = differentiate(expr1, var=var)  # f'
        d_expr2 = differentiate(expr2, var=var)  # g'

        # Build expected result: (f'g - fg')/(g^2)
        numerator = f"(- (* {d_expr1} {expr2}) (* {expr1} {d_expr2}))"
        denominator = f"(^ {expr2} 2)"
        expected = simplify(f"(/ {numerator} {denominator})")

        assert sexp_to_ast(quotient_derivative) == sexp_to_ast(expected)
    except ValueError:
        # Skip test cases that would result in division by zero
        return


@given(st.integers(min_value=1, max_value=5), st.integers(min_value=1, max_value=5))
def test_division_chain_rule(factor1, factor2):
    """Test division combined with chain rule."""
    # d/dx((ax)/(bx)) where a,b are constants
    expr = f"(/ (* {factor1} x) (* {factor2} x))"
    result = differentiate(expr)

    assert exprs_equivalent(
        result, "0", test_values=[1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 314159, -314159]
    )


def test_division_invalid():
    """Test invalid division expressions."""
    # Division by zero
    with pytest.raises(ValueError):
        differentiate("(/ x 0)")

    # Too few arguments
    with pytest.raises(ValueError):
        differentiate("(/ x)")

    # Too many arguments
    with pytest.raises(ValueError):
        differentiate("(/ x 1 2)")
