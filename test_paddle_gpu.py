import paddle

print("Version:", paddle.__version__)
print("CUDA:", paddle.device.is_compiled_with_cuda())
print("Device:", paddle.device.get_device())

x = paddle.randn([1000, 1000])
y = paddle.matmul(x, x)

print("Success!")
print(y.shape)