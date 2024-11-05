import pytest

from symbolic_diff.parser import ASTNode, ast_to_sexp, parse, sexp_to_ast
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


def test_astnode_repr():
    """Test ASTNode.__repr__"""
    node = ASTNode("NUMBER", "42")
    assert repr(node) == "ASTNode(NUMBER, 42, [])"


def test_astnode_eq_non_astnode():
    """Test ASTNode.__eq__ with non-ASTNode"""
    node = ASTNode("NUMBER", "42")
    assert node != "not a node"


def test_ast_to_sexp_invalid_type():
    """Test ast_to_sexp with invalid node type"""
    node = ASTNode("INVALID", "value")
    with pytest.raises(ValueError, match="Unknown node type: INVALID"):
        ast_to_sexp(node)


def test_parse_unexpected_end():
    """Test parse with unexpected end of input"""
    with pytest.raises(ValueError, match="Unexpected end of input"):
        sexp_to_ast("(")


def test_parse_invalid_operator():
    """Test parse with invalid operator"""
    with pytest.raises(ValueError, match="Expected operator"):
        sexp_to_ast("(123)")


def test_parse_missing_rparen():
    """Test parse with missing closing parenthesis"""
    with pytest.raises(ValueError, match="Missing closing parenthesis"):
        sexp_to_ast("(+ x")


def test_parse_unexpected_token():
    """Test parse with unexpected token"""
    with pytest.raises(ValueError, match="Extra tokens after expression"):
        sexp_to_ast("42 )")
