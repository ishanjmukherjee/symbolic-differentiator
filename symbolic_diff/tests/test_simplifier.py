import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from symbolic_diff.parser import ASTNode
from symbolic_diff.simplifier import prettify_float, simplify, simplify_ast

numbers = st.from_regex(r"^[+-]?(\d+\.?\d*|\.\d+)$").map(str.strip)
variables = st.from_regex(r"^[a-z_][a-z0-9_]*$").map(str.strip)


@st.composite
def ast_nodes(draw, max_depth=3):
    """Strategy to generate AST nodes"""
    if max_depth == 0:
        return draw(
            st.one_of(
                st.builds(ASTNode, st.just("NUMBER"), numbers),
                st.builds(ASTNode, st.just("VARIABLE"), variables),
            )
        )

    operator = draw(st.sampled_from(["+", "*", "-", "/", "^"]))
    num_children = draw(st.integers(min_value=2, max_value=3))
    children = [draw(ast_nodes(max_depth=max_depth - 1)) for _ in range(num_children)]
    return ASTNode("OPERATOR", operator, children)


@given(st.floats(allow_infinity=False, allow_nan=False))
def test_prettify_float_property(num):
    """Property-based test for prettify_float"""
    result = prettify_float(num)
    # Should be convertible back to float
    float_result = float(result)
    assert math.isclose(float_result, num, rel_tol=1e-10)


def test_simplify_invalid_node_type():
    """Test simplify with invalid node type"""
    node = ASTNode("INVALID", "value")
    with pytest.raises(ValueError, match="Unknown node type: INVALID"):
        simplify_ast(node)


@given(st.floats(min_value=-1e10, max_value=1e10))
def test_number_simplification(num):
    """Test that numbers are preserved as-is"""
    node = ASTNode("NUMBER", str(num))
    result = simplify_ast(node)
    assert result == node


@given(variables)
def test_variable_simplification(var):
    """Test that variables are preserved as-is"""
    node = ASTNode("VARIABLE", var)
    result = simplify_ast(node)
    assert result == node


@given(st.lists(numbers, min_size=2, max_size=5))
def test_simplify_addition(nums):
    """Test addition simplification"""
    expr = f"(+ {' '.join(nums)})"
    result = simplify(expr)
    expected = str(sum(float(n) for n in nums))
    assert float(result) == pytest.approx(float(expected))


@given(st.lists(numbers, min_size=2, max_size=5))
def test_simplify_multiplication(nums):
    """Test multiplication simplification"""
    expr = f"(* {' '.join(nums)})"
    result = simplify(expr)
    expected = str(math.prod(float(n) for n in nums))
    assert float(result) == pytest.approx(float(expected))


def test_simplify_basic():
    """Test basic simplification cases"""
    assert simplify("(+ 1 2)") == "3"
    assert simplify("(* 2 3)") == "6"
    assert simplify("(- 5 3)") == "2"
    assert simplify("(/ 6 2)") == "3"
    assert simplify("(^ 2 3)") == "8"


def test_simplify_nested_expressions():
    """Test simplification of nested expressions"""
    assert simplify("(+ (* 2 3) (- 5 2))") == "9"
    assert simplify("(* (+ 1 2) (- 4 1))") == "9"
    assert simplify("(/ (+ 2 4) (* 2 1))") == "3"
    assert simplify("(^ (+ 1 1) (* 2 2))") == "16"


@given(variables)
def test_simplify_variables(var):
    """Test simplification with variables"""
    # Variable plus zero
    assert simplify(f"(+ {var} 0)") == var

    # Variable times one
    assert simplify(f"(* {var} 1)") == var

    # Variable divided by one
    assert simplify(f"(/ {var} 1)") == var

    # Variable to power of one
    assert simplify(f"(^ {var} 1)") == var
