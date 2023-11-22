%%  清空环境变量
warning off             % 关闭报警信息
close all               % 关闭开启的图窗
clear                   % 清空变量
clc                     % 清空命令行
% restoredefaultpath
%% 导入数据
res=xlsread('数据集.xlsx');

%%  数据分析
num_size = 0.8;                              % 训练集占数据集比例
outdim = 1;                                  % 最后一列为输出
num_samples = size(res, 1);                  % 样本个数
res = res(randperm(num_samples), :);         % 打乱数据集（不希望打乱时，注释该行）
num_train_s = round(num_size * num_samples); % 训练集样本个数
f_ = size(res, 2) - outdim;                  % 输入特征维度

%%  划分训练集和测试集
P_train = res(1: num_train_s, 1: f_)';
T_train = res(1: num_train_s, f_ + 1: end)';
M = size(P_train, 2);

P_test = res(num_train_s + 1: end, 1: f_)';
T_test = res(num_train_s + 1: end, f_ + 1: end)';
N = size(P_test, 2);

%%  数据归一化
[p_train, ps_input] = mapminmax(P_train,0,1);
p_test = mapminmax('apply',P_test,ps_input);

[t_train, ps_output] = mapminmax(T_train,0,1);
t_test = mapminmax('apply',T_test,ps_output);

%% 节点个数
inputnum  = size(p_train, 1); % 输入层节点数
hiddennum = 15;                % 隐藏层节点数
outputnum = size(t_train, 1); % 输出层节点数

%% 构建网络
net = newff(p_train, t_train, hiddennum);

%% 设置训练参数
net.trainParam.epochs     = 50;      % 训练次数
net.trainParam.goal       = 1e-4;     % 目标误差
net.trainParam.lr         = 0.01;     % 学习率
net.trainParam.showWindow = 0;        % 关闭窗口

%%  参数设置
fun = @getObjValue;                                 % 目标函数
dim = inputnum * hiddennum + hiddennum * outputnum + ...
    hiddennum + outputnum;                          % 优化参数个数
lb  = -1 * ones(1, dim);                            % 优化参数目标下限
ub  =  1 * ones(1, dim);                            % 优化参数目标上限
pop = 20;                                            % 数量
Max_iteration = 20;                                 % 最大迭代次数   

%% 优化算法
[Best_score,Best_pos,curve] = SSA(pop, Max_iteration, lb, ub, dim, fun); 

%% 把最优初始阀值权值赋予网络预测
w1 = Best_pos(1 : inputnum * hiddennum);
B1 = Best_pos(inputnum * hiddennum + 1 : inputnum * hiddennum + hiddennum);
w2 = Best_pos(inputnum * hiddennum + hiddennum + 1 : inputnum * hiddennum + hiddennum + hiddennum*outputnum);
B2 = Best_pos(inputnum * hiddennum + hiddennum + hiddennum * outputnum + 1 : ...
    inputnum * hiddennum + hiddennum + hiddennum * outputnum + outputnum);

net.Iw{1, 1} = reshape(w1, hiddennum, inputnum);
net.Lw{2, 1} = reshape(w2, outputnum, hiddennum);
net.b{1}     = reshape(B1, hiddennum, 1);
net.b{2}     = B2';

%% 网络训练
net.trainParam.showWindow = 1;        % 打开窗口
net = train(net, p_train, t_train);
%% BP网络预测
t_sim1 = sim(net, p_train);
t_sim2 = sim(net, p_test);

%% 数据反归一化
T_sim1 = mapminmax('reverse',t_sim1,ps_output);
T_sim2 = mapminmax('reverse',t_sim2,ps_output);
%%  均方根误差
error1 = sqrt(sum((T_sim1 - T_train).^2) ./ M);
error2 = sqrt(sum((T_sim2 - T_test ).^2) ./ N);

%% 优化曲线
figure
plot(curve, 'linewidth',1.5);
title('SSA')
xlabel('The number of iterations')
ylabel('Fitness')
grid on;
set(gcf,'color','w')

%%  绘图
%% 测试集结果
figure;
plotregression(T_test,T_sim2,['回归图']);
set(gcf,'color','w')
figure;
ploterrhist(T_test-T_sim2,['误差直方图']);
set(gcf,'color','w')
%%  均方根误差 RMSE
error1 = sqrt(sum((T_sim1 - T_train).^2)./M);
error2 = sqrt(sum((T_test - T_sim2).^2)./N);

%%
%决定系数
R1 = 1 - norm(T_train - T_sim1)^2 / norm(T_train - mean(T_train))^2;
R2 = 1 - norm(T_test -  T_sim2)^2 / norm(T_test -  mean(T_test ))^2;

%%
%均方误差 MSE
mse1 = sum((T_sim1 - T_train).^2)./M;
mse2 = sum((T_sim2 - T_test).^2)./N;
%%
%RPD 剩余预测残差
SE1=std(T_sim1-T_train);
RPD1=std(T_train)/SE1;

SE=std(T_sim2-T_test);
RPD2=std(T_test)/SE;
%% 平均绝对误差MAE
MAE1 = mean(abs(T_train - T_sim1));
MAE2 = mean(abs(T_test - T_sim2));
%% 平均绝对百分比误差MAPE
MAPE1 = mean(abs((T_train - T_sim1)./T_train));
MAPE2 = mean(abs((T_test - T_sim2)./T_test));
%%  训练集绘图
figure
plot(1:M,T_train,'r-*',1:M,T_sim1,'b-o','LineWidth',1.5)
legend('真实值','SSA-BP预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'训练集预测结果对比';['(R^2 =' num2str(R1) ' RMSE= ' num2str(error1) ' MSE= ' num2str(mse1) ' RPD= ' num2str(RPD1) ')' ]};
title(string)
set(gcf,'color','w')
%% 预测集绘图
figure
plot(1:N,T_test,'r-*',1:N,T_sim2,'b-o','LineWidth',1.5)
legend('真实值','SSA-BP预测值')
xlabel('预测样本')
ylabel('预测结果')
string={'测试集预测结果对比';['(R^2 =' num2str(R2) ' RMSE= ' num2str(error2)  ' MSE= ' num2str(mse2) ' RPD= ' num2str(RPD2) ')']};
title(string)
set(gcf,'color','w')

%% 测试集误差图
figure  
ERROR3=T_test-T_sim2;
plot(T_test-T_sim2,'b-*','LineWidth',1.5)
xlabel('测试集样本编号')
ylabel('预测误差')
title('测试集预测误差')
grid on;
legend('SSA-BP预测输出误差')
set(gcf,'color','w')
%% 绘制线性拟合图
%% 训练集拟合效果图
figure
plot(T_train,T_sim1,'*r');
xlabel('真实值')
ylabel('预测值')
string = {'训练集效果图';['R^2_c=' num2str(R1)  '  RMSEC=' num2str(error1) ]};
title(string)
hold on ;h=lsline;
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')
%% 预测集拟合效果图
figure
plot(T_test,T_sim2,'ob');
xlabel('真实值')
ylabel('预测值')
string1 = {'测试集效果图';['R^2_p=' num2str(R2)  '  RMSEP=' num2str(error2) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')
%% 求平均
R3=(R1+R2)./2;
error3=(error1+error2)./2;
%% 总数据线性预测拟合图
tsim=[T_sim1,T_sim2]';
S=[T_train,T_test]';
figure
plot(S,tsim,'ob');
xlabel('真实值')
ylabel('预测值')
string1 = {'所有样本拟合预测图';['R^2_p=' num2str(R3)  '  RMSEP=' num2str(error3) ]};
title(string1)
hold on ;h=lsline();
set(h,'LineWidth',1,'LineStyle','-','Color',[1 0 1])
set(gcf,'color','w')
%% 打印出评价指标
disp(['-----------------------误差计算--------------------------'])
disp(['评价结果如下所示：'])
disp(['平均绝对误差MAE为：',num2str(MAE2)])
disp(['均方误差MSE为：       ',num2str(mse2)])
disp(['均方根误差RMSEP为：  ',num2str(error2)])
disp(['决定系数R^2为：  ',num2str(R2)])
disp(['剩余预测残差RPD为：  ',num2str(RPD2)])
disp(['平均绝对百分比误差MAPE为：  ',num2str(MAPE2)])
grid