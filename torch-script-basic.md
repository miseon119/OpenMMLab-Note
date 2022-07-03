# Torchscript Basic

## 模型转换
作为模型部署的一个范式，通常我们都需要生成一个模型的中间表示（IR），这个 IR 拥有相对固定的图结构，所以更容易优化，让我们看一个例子：

```python
import torch 
from torchvision.models import resnet18 
 
# 使用PyTorch model zoo中的resnet18作为例子 
model = resnet18() 
model.eval() 
 
# 通过trace的方法生成IR需要一个输入样例 
dummy_input = torch.rand(1, 3, 224, 224)
# print("dummy_input= ", dummy_input) 
 
# IR生成 
with torch.no_grad(): 
    jit_model = torch.jit.trace(model, dummy_input) 

jit_layer1 = jit_model.layer1 
print(jit_layer1.graph)
print('----------------------------')
print(jit_layer1.code) 
```

output:
```
root@b989fd5b7156:/workdir/openMMLab-Practise# python3 example1.py 
graph(%self.11 : __torch__.torch.nn.modules.container.Sequential,
      %4 : Float(1, 64, 56, 56, strides=[200704, 3136, 56, 1], requires_grad=0, device=cpu)):
  %_1.1 : __torch__.torchvision.models.resnet.___torch_mangle_10.BasicBlock = prim::GetAttr[name="1"](%self.11)
  %_0.1 : __torch__.torchvision.models.resnet.BasicBlock = prim::GetAttr[name="0"](%self.11)
  %6 : Tensor = prim::CallMethod[name="forward"](%_0.1, %4)
  %7 : Tensor = prim::CallMethod[name="forward"](%_1.1, %6)
  return (%7)

----------------------------
def forward(self,
    argument_1: Tensor) -> Tensor:
  _1 = getattr(self, "1")
  _0 = getattr(self, "0")
  _2 = (_1).forward((_0).forward(argument_1, ), )
  return _2
```

> TorchScript 的 IR 是可以还原成 python 代码的，如果你生成了一个 TorchScript 模型并且想知道它的内容对不对，那么可以通过这样的方式来做一些简单的检查。

> 例子中我们使用 trace 的方法生成IR。除了 trace 之外，PyTorch 还提供了另一种生成 TorchScript 模型的方法：script。


```python
# 调用inline pass，对graph做变换 
torch._C._jit_pass_inline(jit_layer1.graph) 
print(jit_layer1.code) 
```
output
```
def forward(self,
    argument_1: Tensor) -> Tensor:
  _1 = getattr(self, "1")
  _0 = getattr(self, "0")
  bn2 = _0.bn2
  conv2 = _0.conv2
  relu = _0.relu
  bn1 = _0.bn1
  conv1 = _0.conv1
  weight = conv1.weight
  input = torch._convolution(argument_1, weight, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
  running_var = bn1.running_var
  running_mean = bn1.running_mean
  bias = bn1.bias
  weight0 = bn1.weight
  input0 = torch.batch_norm(input, weight0, bias, running_mean, running_var, False, 0.10000000000000001, 1.0000000000000001e-05, True)
  input1 = torch.relu_(input0)
  weight1 = conv2.weight
  input2 = torch._convolution(input1, weight1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
  running_var0 = bn2.running_var
  running_mean0 = bn2.running_mean
  bias0 = bn2.bias
  weight2 = bn2.weight
  out = torch.batch_norm(input2, weight2, bias0, running_mean0, running_var0, False, 0.10000000000000001, 1.0000000000000001e-05, True)
  input3 = torch.add_(out, argument_1)
  input4 = torch.relu_(input3)
  bn20 = _1.bn2
  conv20 = _1.conv2
  relu0 = _1.relu
  bn10 = _1.bn1
  conv10 = _1.conv1
  weight3 = conv10.weight
  input5 = torch._convolution(input4, weight3, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
  running_var1 = bn10.running_var
  running_mean1 = bn10.running_mean
  bias1 = bn10.bias
  weight4 = bn10.weight
  input6 = torch.batch_norm(input5, weight4, bias1, running_mean1, running_var1, False, 0.10000000000000001, 1.0000000000000001e-05, True)
  input7 = torch.relu_(input6)
  weight5 = conv20.weight
  input8 = torch._convolution(input7, weight5, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1, False, False, True, True)
  running_var2 = bn20.running_var
  running_mean2 = bn20.running_mean
  bias2 = bn20.bias
  weight6 = bn20.weight
  out0 = torch.batch_norm(input8, weight6, bias2, running_mean2, running_var2, False, 0.10000000000000001, 1.0000000000000001e-05, True)
  input9 = torch.add_(out0, input4)
  return torch.relu_(input9)
```

## 序列化

```python
import torch 
from torchvision.models import resnet18 
 
# 使用PyTorch model zoo中的resnet18作为例子 
model = resnet18() 
model.eval() 
 
# 通过trace的方法生成IR需要一个输入样例 
dummy_input = torch.rand(1, 3, 224, 224)
# print("dummy_input= ", dummy_input) 
 
# IR生成 
with torch.no_grad(): 
    jit_model = torch.jit.trace(model, dummy_input) 

jit_layer1 = jit_model.layer1 
print(jit_layer1.graph)

# 将模型序列化 
jit_model.save('jit_model.pth') 
# 加载序列化后的模型 
jit_model = torch.jit.load('jit_model.pth') 
```

PyTorch 提供了可以用于 TorchScript 模型推理的 c++ API
```cpp
// 加载生成的torchscript模型 
auto module = torch::jit::load('jit_model.pth'); 
// 根据任务需求读取数据 
std::vector<torch::jit::IValue> inputs = ...; 
// 计算推理结果 
auto output = module.forward(inputs).toTensor(); 
```