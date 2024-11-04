import string
import pytest
from hypothesis import given, strategies as st
from symbolic_diff.tokenizer import tokenize, Token

# Helper Hypothesis strategies
operators = st.sampled_from("+-*/^")
# A number may begin with either a '+' or a '-', and be either of the form
# ".500" or "001.500", i.e., begin with a floating point period or have one in
# the middle. str.strip() removes whitespace the strategy may add.
numbers = st.from_regex(r"^[+-]?(\d+\.?\d*|\.\d+)$").map(str.strip)
variables = st.from_regex(r"^[a-z_][a-z0-9_]*$")


def test_number_then_variable():
    """Test that a number followed by letters is correctly separated into a
    number token and a variable token."""
    test_cases = {
        "1x": [Token("NUMBER", "1"), Token("VARIABLE", "x")],
        "11xx": [Token("NUMBER", "11"), Token("VARIABLE", "xx")],
        "123abc": [Token("NUMBER", "123"), Token("VARIABLE", "abc")],
    }

    for expr, expected in test_cases.items():
        tokens = tokenize(expr)
        assert (
            tokens == expected
        ), f"Failed for {expr}: got {tokens}, expected {expected}"


def test_basic_tokenization():
    """Test that a simple expression is tokenized correctly"""
    expr = "(+ x 6)"
    tokens = tokenize(expr)
    expected = [
        Token("LPAREN", "("),
        Token("OPERATOR", "+"),
        Token("VARIABLE", "x"),
        Token("NUMBER", "6"),
        Token("RPAREN", ")"),
    ]
    assert tokens == expected


def test_nested_expression():
    """Test that a nested expression is tokenized correctly"""
    expr = "(+ (* x 2) 6)"
    tokens = tokenize(expr)
    expected = [
        Token("LPAREN", "("),
        Token("OPERATOR", "+"),
        Token("LPAREN", "("),
        Token("OPERATOR", "*"),
        Token("VARIABLE", "x"),
        Token("NUMBER", "2"),
        Token("RPAREN", ")"),
        Token("NUMBER", "6"),
        Token("RPAREN", ")"),
    ]
    assert tokens == expected


@given(st.text(alphabet="xy", min_size=1, max_size=10))
def test_variable_tokenization(var):
    """Test that variable names are tokenized correctly"""
    tokens = tokenize(var)
    assert len(tokens) == 1
    assert tokens[0] == Token("VARIABLE", var)


@given(operators)
def test_operator_tokenization(op):
    """Test that operators are tokenized correctly"""
    tokens = tokenize(op)
    assert len(tokens) == 1
    assert tokens[0] == Token("OPERATOR", op)


@given(numbers)
def test_number_tokenization(num):
    """Test that numbers (including decimals and negatives) are tokenized correctly"""
    tokens = tokenize(num)
    assert len(tokens) == 1
    assert tokens[0] == Token("NUMBER", num)


@given(st.lists(st.text(alphabet=" \t\n", min_size=1), min_size=0, max_size=5))
def test_whitespace_handling(spaces):
    """Test that the tokenizer ignores whitespace"""
    expr = "(".join(spaces)
    tokens = tokenize(expr)
    # Only the left parens should be tokenized, not the whitespace.
    # Since tokens gets rid of whitespace, its length should be (the length of
    # the expression) - (the number of whitespace chars).
    assert len(tokens) == len(expr) - sum(len(s) for s in spaces)
    assert all(t.type == "LPAREN" for t in tokens)


@given(
    st.text(
        alphabet=string.ascii_letters + string.digits + "()+-*/^",
        min_size=0,
        max_size=20,
    )
)
def test_tokenizer_doesnt_crash(expr):
    """Test that tokenizer either succeeds or raises ValueError for any input"""
    try:
        tokens = tokenize(expr)
        # If tokenization succeeds, verify basic properties
        assert all(
            isinstance(t, Token) for t in tokens
        ), f"Non-Token object found in output: {[type(t) for t in tokens]}"
        assert all(
            t.type in {"LPAREN", "RPAREN", "OPERATOR", "NUMBER", "VARIABLE"}
            for t in tokens
        )
    except ValueError:
        # ValueError is the only acceptable exception
        pass


@given(
    st.lists(
        st.tuples(
            st.sampled_from(["LPAREN", "RPAREN", "OPERATOR", "NUMBER", "VARIABLE"]),
            st.text(min_size=1, max_size=5),
        )
    )
)
def test_token_equality(token_pairs):
    """Test Token equality comparison for various token types and values."""
    tokens = [Token(type_, value) for type_, value in token_pairs]

    # Test self-equality
    for token in tokens:
        assert token == token, f"Token failed self-equality check: {token}"

    # Test equality with identical tokens
    for type_, value in token_pairs:
        t1 = Token(type_, value)
        t2 = Token(type_, value)
        assert t1 == t2, f"Tokens with identical attributes not equal: {t1} != {t2}"

    # Test inequality with different tokens
    for i, t1 in enumerate(tokens):
        for j, t2 in enumerate(tokens):
            if i != j:
                expected = token_pairs[i] == token_pairs[j]
                actual = t1 == t2
                assert actual == expected, (
                    f"Token equality mismatch: {t1} vs {t2} "
                    f"(expected {expected}, got {actual})"
                )


def test_token_inequality_with_raw_values():
    """Test that Token objects are not equal to non-Token values"""
    token = Token("NUMBER", "42")
    assert token != "42"
    assert token != ("NUMBER", "42")
