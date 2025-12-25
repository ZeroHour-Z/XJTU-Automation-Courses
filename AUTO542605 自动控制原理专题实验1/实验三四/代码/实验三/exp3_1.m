K = 100;
num = 1;
den = conv(conv([1 1], [0.5 1]),[0.002 1]);
Gp = tf(num * K, den);

[Gm, Pm, Wcg, Wcp] = margin(Gp);
fprintf('未校正相位裕度 Pm: %.4f 度\n', Pm);
fprintf('未校正截止频率 Wcp: %.4f rad/s\n', Wcp);

Pm_c = 50 - Pm + 8;

a = (1 - sin(Pm_c * pi / 180)) / (1 + sin(Pm_c * pi / 180));
Lg = -10 * log10(1/a);
[mag, pha, w] = bode(Gp);

mag_dB = 20 * log10(mag(:));

wmin = w(find(mag_dB >= Lg)); wmin1 = max(wmin);
wmax = w(find(mag_dB <= Lg)); wmax1 = min(wmax);
Wc_new = (wmin1 + wmax1) / 2;

T = 1 / (Wc_new * sqrt(a));
T1 = a * T;
Gc = tf([T 1], [T1 1]);

G_new = Gc * Gp;
[Gm_new, Pm_new, Wcg_new, Wcp_new] = margin(G_new);

fprintf('超前校正传递函数 Gc(s):\n');
fprintf('Gc(s) = (%.6f*s + 1) / (%.6f*s + 1)\n', T, T1);

fprintf('校正参数:\n');
fprintf('a: %.4f\n', a);
fprintf('T: %.6f\n', T);
fprintf('T1: %.6f\n', T1);

fprintf('校正后相位裕度 Pm_new: %.4f 度\n', Pm_new);
if Pm_new >= 50
    disp('相位裕度满足设计要求 (>= 50 度)');
else
    disp('相位裕度不满足设计要求 (< 50 度)');
end
% fprintf('校正后的 K 增益为: %.0f (满足要求 K >= 100)。\n', K);

Gcl_new = feedback(G_new, 1);
Gcl_old = feedback(Gp, 1);

figure(1);
bode(Gp, G_new);
grid on;
legend('未校正', '校正后', 'Location', 'southwest');
title('校正前后的伯德图');

figure(2);
step(Gcl_old, Gcl_new);
grid on;
legend('未校正闭环', '校正后闭环', 'Location', 'southeast');
title('校正前后的阶跃响应');


