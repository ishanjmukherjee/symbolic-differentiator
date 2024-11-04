from hypothesis import given, strategies as st
import pytest
from symbolic_diff.differentiator import differentiate, ast_to_sexp, sexp_to_ast

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
    operator = draw(st.sampled_from(["+"]))  # Starting with +, add other ops later
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
    expected = f"(+ {' '.join(expected_terms)})"
    assert result == expected


@given(st.lists(variables, min_size=2, max_size=5))
def test_addition_multiple_variables(vars):
    """Test derivative of sum of multiple variables."""
    expr = f"(+ {' '.join(vars)})"
    target_var = vars[0]  # Use first variable as target
    result = differentiate(expr, var=target_var)

    # Non-simplified result: "1" if variables match (e.g., differentiating "x"
    # w.r.t. "x") else "0" (e.g., differentiating "y" w.r.t. "x")
    expected_terms = [("1" if v == target_var else "0") for v in vars]
    expected = f"(+ {' '.join(expected_terms)})"

    assert result == expected


@given(sexprs(max_depth=2), sexprs(max_depth=2), variables)
def test_sum_rule(expr1, expr2, var):
    """Property-based test for d/dx(f + g) = d/dx(f) + d/dx(g)"""
    sum_expr = f"(+ {expr1} {expr2})"
    sum_derivative = differentiate(sum_expr, var=var)

    part1 = differentiate(expr1, var=var)
    part2 = differentiate(expr2, var=var)
    separate_derivative = f"(+ {part1} {part2})"

    # Convert both to AST before comparing
    assert sexp_to_ast(sum_derivative) == sexp_to_ast(separate_derivative)
