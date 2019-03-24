# %%
import torch
import matplotlib.pyplot as plt


# %%
# fake data
x = torch.unsqueeze(torch.linspace(-1, 1, 100),
                    dim=1)  # x data (tensor), shape=
# (100, 1)
y = x.pow(2) + 0.2 * torch.randn(x.size())


# %%
def save():
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )

    optimizer = torch.optim.SGD(net1.parameters(), lr=0.5)
    loss_func = torch.nn.MSELoss()

    for i in range(100):
        predict = net1(x)
        loss = loss_func(predict, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # plot result
    plt.figure(1, figsize=(10, 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), predict.data.numpy(), 'r-', lw=5)

    # 2 ways to save the net
    torch.save(net1, 'net1.pkl')  # save entire net
    torch.save(net1.state_dict(), 'net1_params.pkl')  # save only the parameters


# %%
def restore_net():
    net2 = torch.load('net1.pkl')
    predict = net2(x)

    plt.figure(1, figsize=(10, 3))
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), predict.data.numpy(), 'r-', lw=5)


# %%
def restore_params():
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1)
    )


    net3.load_state_dict(torch.load('net1_params.pkl'))
    predict = net3(x)
    plt.figure(1, figsize=(10, 3))
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), predict.data.numpy(), 'r-', lw=5)
    plt.show()


# %%
save()
restore_net()
restore_params()
