import ast
try:
    with open("app.py", "r", encoding="utf-8") as f:
        ast.parse(f.read())
    print("Syntax OK")
except SyntaxError as e:
    print(f"Syntax Error at line {e.lineno}: {e.msg}")
