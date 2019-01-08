#####muti-gpus single-gpu cpu

#training 
# model= net() define the network framework
if use_gpu and len(device_ids)>1:#多gpu训练
    model = model.cuda(device_ids[0])
    model = nn.DataParallel(model, device_ids=device_ids)
# ... 
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.001)
    optimizer = nn.DataParallel(optimizer, device_ids=device_ids)

    optimizer.module.step()


if use_gpu and len(device_ids)==1:#单gpu训练
    model = model.cuda()

model = model   #cpu

## save model 
#save the all model 
