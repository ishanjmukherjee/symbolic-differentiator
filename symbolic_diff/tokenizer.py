class Token:
    def __init__(self, type_, value):
        self.type = type_
        self.value = value

    def __eq__(self, other):
        if not isinstance(other, Token):
            return False
        return self.type == other.type and self.value == other.value

    def __repr__(self):
        return f"Token('{self.type}', '{self.value}')"


def tokenize(expression):
    """Convert a string LISP expression into a sequence of tokens.

    Args:
        expression (str): A LISP-style expression like "(+ x 6)"

    Returns:
        list[Token]: A list of tokens

    Example:
        >>> tokenize("(+ x 6)")
        [Token(LPAREN, '('), Token(OPERATOR, '+'), Token(VARIABLE, 'x'),
         Token(NUMBER, '6'), Token(RPAREN, ')')]
    """
    tokens = []
    current = 0

    while current < len(expression):
        char = expression[current]

        if char.isspace():
            current += 1
            continue

        if char == "(":
            tokens.append(Token("LPAREN", "("))
            current += 1
        elif char == ")":
            tokens.append(Token("RPAREN", ")"))
            current += 1
        elif char.isalpha() or char == "_":  # possible starts of variable
            # Variable
            value = char
            current += 1
            # Absorb the entire (alphanumeric) variable name
            while current < len(expression) and (
                expression[current].isalnum() or expression[current] == "_"
            ):
                value += expression[current]
                current += 1
            tokens.append(Token("VARIABLE", value))
        # Numbers may begin with '-' (for negative), '+' (for positive) or '.'
        # (for float). '-' or '+' is a prefix to a number (and not an operator)
        # if the next token is a number or a floating decimal point
        elif char.isdigit() or (
            (char == "-" or char == "+" or char == ".")
            and current + 1 < len(expression)
            and (expression[current + 1].isdigit() or expression[current + 1] == ".")
        ):
            value = char
            current += 1

            # Absorb the entire number (including the period in floats)
            while current < len(expression) and (
                expression[current].isdigit() or expression[current] == "."
            ):
                value += expression[current]
                current += 1

            # Check that number actually parses as a float, catching exceptions
            # like "0.1.1" and "." while allowing "1."
            try:
                float(value)
            except ValueError:
                raise ValueError(f"'{value}' is not a valid number.")

            tokens.append(Token("NUMBER", value))
        # Check for operators AFTER the check for numbers, since we want to
        # distinguish between '-' and '+' the operator and '-' and '+' the
        # prefixes for number tokens like '-42' and '+42'.
        elif char in "+-*/^":
            tokens.append(Token("OPERATOR", char))
            current += 1
        else:
            raise ValueError(f"Unexpected character: {char}")

    return tokens
