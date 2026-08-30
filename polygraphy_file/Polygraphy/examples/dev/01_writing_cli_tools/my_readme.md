编写自定义命令行工具
简介
Polygraphy 包含了多种辅助工具，使从头开始编写新的命令行工具变得更加容易。
在本例中，我们将编写一个名为 gen-data 的全新工具，该工具将使用 Polygraphy 的默认数据加载器生成随机数据，并将其写入输出文件。用户可以指定要生成的数值数量以及输出路径。
为此，我们将创建一个继承自 Tool 的子类，并使用 Polygraphy 提供的 DataLoaderArgs 参数组。
运行示例:
您可以在当前目录下运行该示例工具。例如：
./gen-data -o data.json --num-values 25

我们甚至可以使用 inspect data 来查看生成的数据：
polygraphy inspect data data.json -s

要查看示例工具中可用的其他命令行选项，
请运行：
./gen-data -h