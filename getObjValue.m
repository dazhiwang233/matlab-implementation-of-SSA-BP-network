function objValue = getObjValue(parameter)
% Ŀ�꺯����һ������ʽ���̣�Ψһ�Ĵ����ǲ�������������������������ΪĿ�꺯����ֵ��
% ����ʾ����һ����������񣬲�����С��RMSE��Ŀ�꺯����
% ������ѵ����������Ҫ��ȡѵ�������Լ���Ӧ�ı�ǩ�������Ŀ�꺯���ڲ���ȡ���ݣ������ַ�ʽ��
% 
% ��1������ѵ�����ݺͱ�ǩ��ȫ�ֱ���
% ��2������load������ȡѵ�����ݺͱ�ǩ
% ��3������evalin������ȡ�������ռ��ѵ�����ݺͱ�ǩ

    % �õ���Ӧѵ������
    p_train = evalin('base', 'p_train');
    t_train = evalin('base', 't_train');
    net = evalin('base', 'net');
    ps_output = evalin('base', 'ps_output');
    
    inputnum  = evalin('base', 'inputnum');
    hiddennum = evalin('base', 'hiddennum');
    outputnum = evalin('base', 'outputnum');
    
    % �õ��Ż�����
    %% �����ų�ʼ��ֵȨֵ��������Ԥ��
    w1 = parameter(1 : inputnum * hiddennum);
    B1 = parameter(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);
    w2 = parameter(inputnum * hiddennum + hiddennum + 1 : inputnum * hiddennum + hiddennum + hiddennum*outputnum);
    B2 = parameter(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
        inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);

    net.Iw{1, 1} = reshape(w1, hiddennum, inputnum);
    net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);
    net.b{1}     = reshape(B1, hiddennum, 1);
    net.b{2}     = B2';

    %% ����ѵ��
    net = train(net,p_train,t_train); 
    
    %% BP����Ԥ��
    t_sim1 = sim(net,p_train);
    
    %% ���ݷ���һ��
    T_sim1 = mapminmax('reverse',t_sim1,ps_output);
    T_train = mapminmax('reverse',t_train,ps_output);
    
    %% ��������
    accuracy = sqrt(sum((T_sim1 - T_train).^2)./length(T_sim1));

    % ��RMSE��Ϊ�Ż���Ŀ�꺯��ֵ
    if size(accuracy, 1) == 0
        objValue = 10000;
    else
        objValue = accuracy(1);
    end
end

