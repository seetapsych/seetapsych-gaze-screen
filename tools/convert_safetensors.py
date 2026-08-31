# 这个文件时间torch.load一个字典，然后保存成safetensors格式，参考代码：
# state_dict = torch.load(state_dict_file, map_location=device)
# safetensors.torch.save_file(state_dict, os.path.splitext(state_dict_file)[0] + ".safetensors")
# 脚本可以输入一个或两个参数，输入一个参数就是加载pth文件，然后替换扩展名作为输出文件。
# 如果两个参数就是输入文件和输出文件。
