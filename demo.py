from symbolic_diff import differentiate

def demo_basic_operations():
    """Demonstrate basic differentiation operations."""
    print("Basic Operations:")
    print("-" * 50)
    
    examples = [
        ("x", "Basic variable"),                    # 1
        ("(+ x 5)", "Addition with constant"),      # 1
        ("(* x 3)", "Multiplication by constant"),  # 3
        ("(^ x 2)", "Square function"),             # (* x 2)
        ("(/ x 2)", "Division by constant"),        # 0.5
        ("(- x 1)", "Subtraction"),                 # 1
    ]
    
    for expr, desc in examples:
        print(f"{desc}:")
        print(f"  Expression: {expr}")
        print(f"  Derivative: {differentiate(expr)}")
        print()

def demo_compound_expressions():
    """Demonstrate differentiation of compound expressions."""
    print("Compound Expressions:")
    print("-" * 50)
    
    examples = [
        ("(* x x)", "x * x"),                       # (+ x x)
        ("(* (+ x 1) (- x 2))", "(x + 1)(x - 2)"),  # (+ (+ x -2) (+ x 1))
        ("(/ (^ x 2) (+ x 1))", "x²/(x + 1)"),      # (/ (+ (* (* x 2) (+ x 1)) 
                                                    #    (* (^ x 2) -1)) (^ (+ x 1) 2))   
        ("(+ (* 2 (^ x 3)) (* -1 x))", "2x³ - x"),  # (+ (* (* (^ x 2) 3) 2) -1)
    ]
    
    for expr, desc in examples:
        print(f"{desc}:")
        print(f"  Expression: {expr}")
        print(f"  Derivative: {differentiate(expr)}")
        print()

def demo_different_variables():
    """Demonstrate differentiation with respect to different variables."""
    print("Different Variables:")
    print("-" * 50)
    
    expression = "(+ (* 2 x) (* y y))"
    print(f"Expression: {expression}")
    print(f"d/dx: {differentiate(expression, 'x')}") # 2
    print(f"d/dy: {differentiate(expression, 'y')}") # (+ y y)
    print()

def demo_error_handling():
    """Demonstrate error handling."""
    print("Error Handling:")
    print("-" * 50)
    
    error_cases = [
        ("(+ x", "Unmatched parenthesis"),      # Error: Missing closing parenthesis
        ("(@ x 2)", "Invalid operator"),        # Error: Unexpected character: @
        ("(/ x 0)", "Division by zero"),        # Error: Cannot divide by zero
    ]
    
    for expr, desc in error_cases:
        print(f"Attempting: {desc}")
        print(f"Expression: {expr}")
        try:
            result = differentiate(expr)
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {str(e)}")
        print()

def main():
    """Run all demos."""
    print("Symbolic Differentiator Demo")
    print("=" * 50)
    print()
    
    demo_basic_operations()
    demo_compound_expressions()
    demo_different_variables()
    demo_error_handling()

if __name__ == "__main__":
    main()