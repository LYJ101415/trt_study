使用自定义输入数据
对于任何使用推理输入数据的工具（如 run 或 convert），Polygraphy
提供了两种提供自定义输入数据的方式：
--load-inputs/--load-input-data，该选项接受一个指向 JSON 文件的路径，文件中包含
List[Dict[str, np.ndarray]] 类型的数据。
该 JSON 文件应使用 polygraphy.json 子模块中的 Polygraphy JSON 工具（如 save_json）创建。
注意：这会导致 Polygraphy 将整个对象加载到内存中，因此如果数据量非常大，
这种方式可能不切实际甚至无法实现。
--data-loader-script，该选项接受一个指向 Python 脚本的路径，该脚本需定义一个返回数据加载器的 load_data 函数。数据加载器可以是任何生成
Dict[str, np.ndarray] 的可迭代对象或生成器。通过使用生成器，我们可以避免一次性加载所有数据，
而是将其限制为每次仅加载单个输入。
提示：如果您现有的脚本中已经定义了此类函数，则无需仅仅为了配合 --data-loader-script
而创建一个单独的脚本。您可以直接使用现有脚本，并在函数名不是 load_data 时指定其名称即可。
延伸阅读
请参阅 `run` 示例 05，
(../examples/cli/run/05_comparing_with_custom_input_data/)
其中包含了上述两种方法的示例。