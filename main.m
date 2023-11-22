%%  ��ջ�������
warning off             % �رձ�����Ϣ
close all               % �رտ�����ͼ��
clear                   % ��ձ���
clc                     % ���������
% restoredefaultpath
%% ��������
res=xlsread('���ݼ�.xlsx');

%%  ���ݷ���
num_size = 0.8;                              % ѵ����ռ���ݼ�����
outdim = 1;                                  % ���һ��Ϊ���
num_samples = size(res, 1);                  % ��������
res = res(randperm(num_samples), :);         % �������ݼ�����ϣ������ʱ��ע�͸��У�
num_train_s = round(num_size * num_samples); % ѵ������������
f_ = size(res, 2) - outdim;                  % ��������ά��

%%  ����ѵ�����Ͳ��Լ�
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  ���ݹ�һ��
[p_train, ps_input] = mapminmax(P_train,0,1);
p_test = mapminmax('apply',P_test,ps_input);

[t_train, ps_output] = mapminmax(T_train,0,1);
t_test = mapminmax('apply',T_test,ps_output);

%% �ڵ����
inputnum  = size(p_train, 1); % �����ڵ���
hiddennum = 15;                % ���ز�ڵ���
outputnum = size(t_train, 1); % �����ڵ���

%% ��������
net = newff(p_train, t_train, hiddennum);

%% ����ѵ������
net.trainParam.epochs     = 50;      % ѵ������
net.trainParam.goal       = 1e-4;     % Ŀ�����
net.trainParam.lr         = 0.01;     % ѧϰ��
net.trainParam.showWindow = 0;        % �رմ���

%%  ��������
fun = @getObjValue;                                 % Ŀ�꺯��
dim = inputnum * hiddennum + hiddennum * outputnum + ...
    hiddennum + outputnum;                          % �Ż���������
lb  = -1 * ones(1, dim);                            % �Ż�����Ŀ������
ub  =  1 * ones(1, dim);                            % �Ż�����Ŀ������
pop = 20;                                            % ����
Max_iteration = 20;                                 % ����������   

%% �Ż��㷨
[Best_score,Best_pos,curve] = SSA(pop, Max_iteration, lb, ub, dim, fun); 

%% �����ų�ʼ��ֵȨֵ��������Ԥ��
w1 = Best_pos(1 : inputnum * hiddennum);
B1 = Best_pos(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);
w2 = Best_pos(inputnum * hiddennum + hiddennum + 1 : inputnum * hiddennum + hiddennum + hiddennum*outputnum);
B2 = Best_pos(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
    inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);

net.Iw{1, 1} = reshape(w1, hiddennum, inputnum);
net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);
net.b{1}     = reshape(B1, hiddennum, 1);
net.b{2}     = B2';

%% ����ѵ��
net.trainParam.showWindow = 1;        % �򿪴���
net = train(net, p_train, t_train);
%% BP����Ԥ��
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test);

%% ���ݷ���һ��
T_sim1 = mapminmax('reverse',t_sim1,ps_output);
T_sim2 = mapminmax('reverse',t_sim2,ps_output);
%%  ���������
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

%% �Ż�����
figure
plot(curve, 'linewidth',1.5);
title('SSA')
xlabel('The number of iterations')
ylabel('Fitness')
grid on;
set(gcf,'color','w')

%%  ��ͼ
%% ���Լ����
figure;
plotregression(T_test,T_sim2,['�ع�ͼ']);
set(gcf,'color','w')
figure;
ploterrhist(T_test-T_sim2,['���ֱ��ͼ']);
set(gcf,'color','w')
%%  ��������� RMSE
error1 = sqrt(sum((T_sim1 - T_train).^2)./M);
error2 = sqrt(sum((T_test - T_sim2).^2)./N);

%%
%����ϵ��
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;

%%
%������� MSE
mse1 = sum((T_sim1 - T_train).^2)./M;
mse2 = sum((T_sim2 - T_test).^2)./N;
%%
%RPD ʣ��Ԥ��в�
SE1=std(T_sim1-T_train);
RPD1=std(T_train)/SE1;

SE=std(T_sim2-T_test);
RPD2=std(T_test)/SE;
%% ƽ���������MAE
MAE1 = mean(abs(T_train - T_sim1));
MAE2 = mean(abs(T_test - T_sim2));
%% ƽ�����԰ٷֱ����MAPE
MAPE1 = mean(abs((T_train - T_sim1)./T_train));
MAPE2 = mean(abs((T_test - T_sim2)./T_test));
%%  ѵ������ͼ
figure
plot(1:M,T_train,'r-*',1:M,T_sim1,'b-o','LineWidth',1.5)
legend('��ʵֵ','SSA-BPԤ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string={'ѵ����Ԥ�����Ա�';['(R^2 =' num2str(R1) ' RMSE= ' num2str(error1) ' MSE= ' num2str(mse1) ' RPD= ' num2str(RPD1) ')' ]};
title(string)
set(gcf,'color','w')
%% Ԥ�⼯��ͼ
figure
plot(1:N,T_test,'r-*',1:N,T_sim2,'b-o','LineWidth',1.5)
legend('��ʵֵ','SSA-BPԤ��ֵ')
xlabel('Ԥ������')
ylabel('Ԥ����')
string={'���Լ�Ԥ�����Ա�';['(R^2 =' num2str(R2) ' RMSE= ' num2str(error2)  ' MSE= ' num2str(mse2) ' RPD= ' num2str(RPD2) ')']};
title(string)
set(gcf,'color','w')

%% ���Լ����ͼ
figure  
ERROR3=T_test-T_sim2;
plot(T_test-T_sim2,'b-*','LineWidth',1.5)
xlabel('���Լ��������')
ylabel('Ԥ�����')
title('���Լ�Ԥ�����')
grid on;
legend('SSA-BPԤ��������')
set(gcf,'color','w')
%% �����������ͼ
%% ѵ�������Ч��ͼ
figure
plot(T_train,T_sim1,'*r');
xlabel('��ʵֵ')
ylabel('Ԥ��ֵ')
string = {'ѵ����Ч��ͼ';['R^2_c=' num2str(R1)  '  RMSEC=' num2str(error1) ]};
title(string)
hold on ;h=lsline;
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')
%% Ԥ�⼯���Ч��ͼ
figure
plot(T_test,T_sim2,'ob');
xlabel('��ʵֵ')
ylabel('Ԥ��ֵ')
string1 = {'���Լ�Ч��ͼ';['R^2_p=' num2str(R2)  '  RMSEP=' num2str(error2) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')
%% ��ƽ��
R3=(R1+R2)./2;
error3=(error1+error2)./2;
%% ����������Ԥ�����ͼ
tsim=[T_sim1,T_sim2]';
S=[T_train,T_test]';
figure
plot(S,tsim,'ob');
xlabel('��ʵֵ')
ylabel('Ԥ��ֵ')
string1 = {'�����������Ԥ��ͼ';['R^2_p=' num2str(R3)  '  RMSEP=' num2str(error3) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')
%% ��ӡ������ָ��
disp(['-----------------------������--------------------------'])
disp(['���۽��������ʾ��'])
disp(['ƽ���������MAEΪ��',num2str(MAE2)])
disp(['�������MSEΪ��       ',num2str(mse2)])
disp(['���������RMSEPΪ��  ',num2str(error2)])
disp(['����ϵ��R^2Ϊ��  ',num2str(R2)])
disp(['ʣ��Ԥ��в�RPDΪ��  ',num2str(RPD2)])
disp(['ƽ�����԰ٷֱ����MAPEΪ��  ',num2str(MAPE2)])
grid