import pytest

from symbolic_diff.parser import ASTNode, parse
from symbolic_diff.tokenizer import tokenize


def test_basic_parsing():
    """Test that a simple expression is parsed correctly"""
    tokens = tokenize("(+ x 6)")
    ast = parse(tokens)
    expected = ASTNode(
        "OPERATOR", "+", [ASTNode("VARIABLE", "x", []), ASTNode("NUMBER", "6", [])]
    )
    assert ast == expected


def test_nested_parsing():
    """Test that a nested expression is parsed correctly"""
    tokens = tokenize("(+ (* x 2) 6)")
    ast = parse(tokens)
    expected = ASTNode(
        "OPERATOR",
        "+",
        [
            ASTNode(
                "OPERATOR",
                "*",
                [ASTNode("VARIABLE", "x", []), ASTNode("NUMBER", "2", [])],
            ),
            ASTNode("NUMBER", "6", []),
        ],
    )
    assert ast == expected


def test_malformed_expression():
    """Test that a malformed expression raises a ValueError"""
    with pytest.raises(ValueError):
        # Missing closing parenthesis
        parse(tokenize("(+ x"))
        # Missing opening parenthesis
        parse(tokenize("+ x 1)"))
        parse(tokenize("+"))
        parse(tokenize("+ x"))
