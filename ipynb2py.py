import nbformat
from nbconvert import PythonExporter
from pathlib import Path

def ipynb_to_py(ipynb_path: str, py_path: str | None = None):
    ipynb_path = Path(ipynb_path)
    if py_path is None:
        py_path = ipynb_path.with_suffix(".py")
    else:
        py_path = Path(py_path)

    # 读取 ipynb
    nb = nbformat.read(ipynb_path, as_version=4)

    # 用 nbconvert 转成 python 源码字符串
    exporter = PythonExporter()
    body, _ = exporter.from_notebook_node(nb)

    # 写入 .py 文件
    py_path.write_text(body, encoding="utf-8")
    print(f"已生成: {py_path}")

if __name__ == "__main__":
    # 示例：把当前目录下的 infer.ipynb 转成 infer.py
    ipynb_to_py("infer2.ipynb")