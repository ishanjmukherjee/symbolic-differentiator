import pytest
from hypothesis import given
from hypothesis import strategies as st

from symbolic_diff.differentiator import ast_to_sexp, differentiate, sexp_to_ast

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
    operator = draw(st.sampled_from(["+", "*", "^"]))
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
def test_sum_rule_property(expr1, expr2, var):
    """Property-based test for d/dx(f + g) = d/dx(f) + d/dx(g)"""
    sum_expr = f"(+ {expr1} {expr2})"
    sum_derivative = differentiate(sum_expr, var=var)

    part1 = differentiate(expr1, var=var)
    part2 = differentiate(expr2, var=var)
    separate_derivative = f"(+ {part1} {part2})"

    # Convert both to AST before comparing
    assert sexp_to_ast(sum_derivative) == sexp_to_ast(separate_derivative)


@given(st.lists(numbers, min_size=2, max_size=5))
def test_product_constants(nums):
    """Test that the derivative of a product of constants is 0."""
    expr = f"(* {' '.join(nums)})"
    result = differentiate(expr)
    # Should be a sum of terms, each containing a 0
    root = sexp_to_ast(result)
    assert (
        root.value == "+"
        and root.type == "OPERATOR"
        and (child.value == "0" and child.type == "NUMBER" for child in root.children)
    )


@given(variables, numbers)
def test_product_rule_basic(var, num):
    """Test basic product rule with variable and constant."""
    expr = f"(* {var} {num})"
    result = differentiate(expr, var=var)
    expected = f"(+ (* 1 {num}) (* {var} 0))"
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
    expected = f"(+ (* {d_expr1} {expr2}) (* {expr1} {d_expr2}))"

    assert sexp_to_ast(product_derivative) == sexp_to_ast(expected)


@given(st.lists(variables, min_size=2, max_size=5))
def test_product_rule_variables(vars):
    """Test basic properties of product rule with multiple variables."""
    expr = f"(* {' '.join(vars)})"
    target_var = vars[0]
    result = differentiate(expr, var=target_var)
    # Verify that the derivative is a sum
    ast = sexp_to_ast(result)
    assert ast.value == "+"
    assert len(ast.children) == len(vars)


def test_power_rule_basic():
    """Test basic power rule cases."""
    # x^0 -> 0
    assert differentiate("(^ x 0)") == "0"

    # x^1 -> 1
    assert differentiate("(^ x 1)") == "1"

    # x^2 -> 2x^1
    result = differentiate("(^ x 2)")
    expected = "(* 2 (^ x 1) 1)"
    expected_simplified = "(* 2 (^ x 1))"
    assert result == expected or result == expected_simplified
    assert sexp_to_ast(result) == sexp_to_ast(expected) or sexp_to_ast(
        result
    ) == sexp_to_ast(
        expected_simplified
    )  # simplified or unsimplified form

    # x^3 -> 3x^2
    result = differentiate("(^ x 3)")
    expected = "(* 3 (^ x 2) 1)"
    expected_simplified = "(* 3 (^ x 2))"
    assert result == expected or result == expected_simplified
    assert sexp_to_ast(result) == sexp_to_ast(expected) or sexp_to_ast(
        result
    ) == sexp_to_ast(expected_simplified)


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
        expected = f"(* {n} (^ x {n-1}))"
        expected_simplified = f"(* {n} (^ x {n-1}) 1)"
        assert result == expected or result == expected_simplified
        assert sexp_to_ast(result) == sexp_to_ast(expected) or sexp_to_ast(
            result
        ) == sexp_to_ast(expected_simplified)


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
        expected = f"(* {n} (^ {var} {n-1}) 1)"
        expected_simplified = f"(* {n} (^ {var} {n-1}))"
        assert result == expected or result == expected_simplified
        assert sexp_to_ast(result) == sexp_to_ast(expected) or sexp_to_ast(
            result
        ) == sexp_to_ast(expected_simplified)


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
    expected = (
        f"(* {exponent} (^ (* {factor} x) {exponent-1}) (+ (* 0 x) (* {factor} 1)))"
    )
    expected_simplified = f"(* {exponent} (^ (* {factor} x) {exponent-1}) {factor})"
    assert result == expected or result == expected_simplified
    assert sexp_to_ast(result) == sexp_to_ast(expected) or sexp_to_ast(
        result
    ) == sexp_to_ast(expected_simplified)


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
