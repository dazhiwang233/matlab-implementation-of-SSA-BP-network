function objValue = getObjValue(parameter)
% 目标函数是一个非显式过程，唯一的传参是参数（或参数向量），输出参数为目标函数的值，
% 由于示例是一个多分类任务，采用最小化RMSE的目标函数。
% 由于在训练过程中需要读取训练数据以及对应的标签，因此在目标函数内部读取数据，有三种方式：
% 
% （1）定义训练数据和标签的全局变量
% （2）利用load函数读取训练数据和标签
% （3）利用evalin函数读取主函数空间的训练数据和标签

    % 得到相应训练数据
    p_train = evalin('base', 'p_train');
    t_train = evalin('base', 't_train');
    net = evalin('base', 'net');
    ps_output = evalin('base', 'ps_output');
    
    inputnum  = evalin('base', 'inputnum');
    hiddennum = evalin('base', 'hiddennum');
    outputnum = evalin('base', 'outputnum');
    
    % 得到优化参数
    %% 把最优初始阀值权值赋予网络预测
    w1 = parameter(1 : inputnum * hiddennum);
    B1 = parameter(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);
    w2 = parameter(inputnum * hiddennum + hiddennum + 1 : inputnum * hiddennum + hiddennum + hiddennum*outputnum);
    B2 = parameter(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
        inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);

    net.Iw{1, 1} = reshape(w1, hiddennum, inputnum);
    net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);
    net.b{1}     = reshape(B1, hiddennum, 1);
    net.b{2}     = B2';

    %% 网络训练
    net = train(net,p_train,t_train); 
    
    %% BP网络预测
    t_sim1 = sim(net,p_train);
    
    %% 数据反归一化
    T_sim1 = mapminmax('reverse',t_sim1,ps_output);
    T_train = mapminmax('reverse',t_train,ps_output);
    
    %% 性能评价
    accuracy = sqrt(sum((T_sim1 - T_train).^2)./length(T_sim1));

    % 以RMSE作为优化的目标函数值
    if size(accuracy, 1) == 0
        objValue = 10000;
    else
        objValue = accuracy(1);
    end
end

