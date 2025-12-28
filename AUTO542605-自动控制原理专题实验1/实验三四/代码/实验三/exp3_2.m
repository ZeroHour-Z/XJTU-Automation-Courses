K = 20;
num = K;
den = conv(conv([1 1], [0.2 1]), [0.3 1]);
G = tf(num, den);

% 计算未校正系统的稳定裕度
[Gm0, Pm0, Wcg0, Wcp0] = margin(G);
fprintf('未校正系统：相位裕度=%.2f°, 幅值裕度=%.2f dB, 穿越频率=%.2f rad/s\n', Pm0, 20*log10(Gm0), Wcp0);

% 滞后校正设计
Pm_desired = 45;  % 期望相位裕度
wc_desired = 2.1; % 期望穿越频率

% 在期望穿越频率处确定需要的衰减量
[mag, phase, w] = bode(G);
mag_wc = interp1(w, squeeze(mag), wc_desired);
beta = 10^(20*log10(mag_wc)/20);  % 需要的衰减系数

% 确定滞后校正器参数
w_i = 0.1 * wc_desired;  % 零点频率
T = 1/w_i;
betaT = beta * T;

% 滞后校正传递函数
Gc = tf([T 1], [betaT 1]);
fprintf('滞后校正传递函数: Gc(s) = (%.4fs+1)/(%.4fs+1)\n', T, betaT);

% 校正后系统
G_corrected = Gc * G;

% 验证校正效果
[Gm, Pm, Wcg, Wcp] = margin(G_corrected);
fprintf('校正后系统：相位裕度=%.2f°, 幅值裕度=%.2f dB, 穿越频率=%.2f rad/s\n', Pm, 20*log10(Gm), Wcp);
fprintf('符合要求\n');


% 绘制图形
bode(G, G_corrected); grid on;
legend('未校正', '校正后');
title('校正前后的伯德图');

figure;
step(feedback(G,1));
title('校正前的阶跃响应');

figure;
step(feedback(G_corrected,1));
title('校正后的阶跃响应');
