a0 = 8;
a1 = 12;
a2 = 6;
a3 = 4;

syms K_sym s;

% s4: 1,  a2, a0+K
% s3: a3, a1, 0
% s2: b1, b2, 0
% s1: c1, 0,  0
% s0: d1, 0,  0

b1 = (a3*a2-1*a1)/a3;
b2 = (a3*(a0+K_sym))/a3;
c1 = (b1*a1-a3*b2)/b1;
d1 = b2;

K_values = -10:0.01:10;
stable_K = [];

for K_test = K_values
    b1_val = double(subs(b1, K_sym, K_test));
    c1_val = double(subs(c1, K_sym, K_test));
    d1_val = double(subs(d1, K_sym, K_test));    
    if a3 > 0 && b1_val > 0 && c1_val > 0 && d1_val > 0
        stable_K = [stable_K, K_test];
    end
end
if ~isempty(stable_K)
    K_min = min(stable_K);
    K_max = max(stable_K);
    fprintf('稳定范围: %.2f<K<%.2f\n', K_min, K_max);
    K_stable = (K_max+K_min)/2;      % 稳定系统
    K_unstable = K_max + 5;              % 不稳定系统
    K_critical = K_max;                  % 临界稳定系统
else
    fprintf('未找到稳定的K值范围\n');
    K_stable = 1;
    K_unstable = 10;
    K_critical = 5;
end
fprintf('稳定:%f\n',K_stable);
fprintf('不稳定:%f\n',K_unstable);
fprintf('临界:%f\n',K_critical);
figure('Position', [100, 100, 1200, 800]);

subplot(2,3,1);
G_stable = tf(K_stable, [1, a3, a2, a1, a0+K_stable]);
step(G_stable, 20);
title('稳定系统');
grid on;

subplot(2,3,2);
G_unstable = tf(K_unstable, [1, a3, a2, a1, a0+K_unstable]);
step(G_unstable, 20);
title('不稳定系统');
grid on;

subplot(2,3,3);
G_critical = tf(K_critical, [1, a3, a2, a1, a0+K_critical]);
step(G_critical, 20);
title('临界稳定系统');
grid on;

% 绘制极点分布
subplot(2,3,4);
p_stable = pole(G_stable);
plot(real(p_stable), imag(p_stable), 'bx', 'MarkerSize', 10);
hold on;
plot([0,0], ylim, 'k-');
plot(xlim, [0,0], 'k-');
title('稳定系统极点分布');
grid on;

subplot(2,3,5);
p_unstable = pole(G_unstable);
plot(real(p_unstable), imag(p_unstable), 'bx', 'MarkerSize', 10);
hold on;
plot([0,0], ylim, 'k-');
plot(xlim, [0,0], 'k-');
title('不稳定');
grid on;

subplot(2,3,6);
p_critical = pole(G_critical);
plot(real(p_critical), imag(p_critical), 'bx', 'MarkerSize', 10);
hold on;
plot([0,0], ylim, 'k-');
plot(xlim, [0,0], 'k-');
title('临界稳定');
grid on;