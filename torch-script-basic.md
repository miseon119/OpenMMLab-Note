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
