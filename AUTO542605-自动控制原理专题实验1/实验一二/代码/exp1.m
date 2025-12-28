% 1(1)
w_n = 2;
ei_values = [0, 0.2, 0.6, 1, 1.3];
ei_labels = {'无阻尼 ξ=0', '欠阻尼 ξ=0.2', '欠阻尼 ξ=0.6', '临界阻尼 ξ=1', '过阻尼 ξ=1.3'};
colors = ['r', 'g', 'b', 'y', 'm'];
line_styles = {'-', '-', '-', '-', '-'};
figure('Position', [100, 100, 800, 600]);
t = 0:0.01:10;
hold on;
for i = 1:length(ei_values)
    ei = ei_values(i);
    if ei == 0
        num = w_n^2;
        den = [1, 0, w_n^2];
    else
        num = w_n^2;
        den = [1, 2*ei*w_n, w_n^2];
    end

    sys = tf(num, den);
    [y, t] = step(sys, t);
    plot(t, y, 'Color', colors(i), 'LineStyle', line_styles{i}, 'DisplayName', ei_labels{i});
end
grid on;
xlabel('t/s');
ylabel('振幅');
legend('show', 'Location', 'southeast');
yline(1, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
xlim([0, 10]);
ylim([-0.2, 2.2]);

% 1(2)
ei_1 = 0.6;   % 欠阻尼
ei_2 = 1.3;    % 过阻尼
t = 0:0.01:100;
colors = ['r', 'g', 'b'];
line_styles = {'-', '-', '-'};
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
w_n_values1 = [1/5, 1, 5];
base_w_n = 2;
hold on;
for i = 1:length(w_n_values1)
    w_n = base_w_n * w_n_values1(i);
    num = w_n^2;
    den = [1, 2*ei_1*w_n, w_n^2];
    sys = tf(num, den);
    [y, t] = step(sys, t);
    plot(t, y, 'Color', colors(i), 'LineStyle', line_styles{i}, 'DisplayName', sprintf('ω_n = %.1f', w_n));
end
grid on;
xlabel('t/s');
ylabel('振幅');
legend('show', 'Location', 'southeast');
yline(1, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
xlim([0, 50]);
ylim([0, 1.5]);

subplot(1, 2, 2);
w_n_values2 = [1/10, 1, 10];
hold on;
for i = 1:length(w_n_values2)
    w_n = base_w_n * w_n_values2(i);
    num = w_n^2;
    den = [1, 2*ei_2*w_n, w_n^2];
    sys = tf(num, den);
    [y, t] = step(sys, t);
    plot(t, y, 'Color', colors(i), 'LineStyle', line_styles{i}, 'DisplayName', sprintf('ω_n = %.1f', w_n));
end
grid on;
xlabel('t/s');
ylabel('振幅');
legend('show', 'Location', 'southeast');
yline(1, 'k--', 'HandleVisibility', 'off');
xlim([0, 100]);
ylim([0, 1.5]);

% 1(3)
ei_values = [0, 0.2, 0.6, 1, 1.3];
sigma_percent = [];
tp = []; tr = []; ts = []; ess = [];
for i = 1:length(ei_values)
    ei = ei_values(i); wn = 2; num = wn^2; den = [1, 2*ei*wn, wn^2];
    sys = tf(num, den);
    t = 0:0.01:20;
    [y, t] = step(sys, t); yss = dcgain(sys);
    upper_bound = yss * 1.02; lower_bound = yss * 0.98;
    [ymax, idx_peak] = max(y);
    sigma_percent = [sigma_percent, (ymax - yss) / yss * 100];
    tp = [tp, t(idx_peak)];
    idx_10 = find(y >= 0.1*yss, 1); idx_90 = find(y >= 0.9*yss, 1);
    tr = [tr, t(idx_90) - t(idx_10)];
    idx_settle = find(y < lower_bound | y > upper_bound, 1, 'last');
    if isempty(idx_settle)
        ts = [ts, 0];
    else
        ts = [ts, t(idx_settle)];
    end
    ess = [ess, abs(1 - yss)];
end
fprintf('超调量 σ%%:'); disp(sigma_percent);
fprintf('峰值时间 tp:'); disp(tp);
fprintf('上升时间 tr:'); disp(tr);
fprintf('调节时间 ts:'); disp(ts);
fprintf('稳态误差 ess:'); disp(ess);

wn_values = [0.4, 2, 10];
sigma_percent = [];
tp = []; tr = []; ts = []; ess = [];
for i = 1:length(wn_values)
    ei = 0.6; wn = wn_values(i); num = wn^2; den = [1, 2*ei*wn, wn^2];
    sys = tf(num, den);
    t = 0:0.01:20;
    [y, t] = step(sys, t); yss = dcgain(sys);
    [ymax, idx_peak] = max(y);
    sigma_percent = [sigma_percent, (ymax - yss) / yss * 100];
    tp = [tp, t(idx_peak)];
    idx_10 = find(y >= 0.1*yss, 1); idx_90 = find(y >= 0.9*yss, 1);
    tr = [tr, t(idx_90) - t(idx_10)];
    idx_settle = find(y < lower_bound | y > upper_bound, 1, 'last');
    if isempty(idx_settle)
        ts = [ts, 0];
    else
        ts = [ts, t(idx_settle)];
    end
    ess = [ess, abs(1 - yss)];
end
fprintf('超调量 σ%%:'); disp(sigma_percent);
fprintf('峰值时间 tp:'); disp(tp);
fprintf('上升时间 tr:'); disp(tr);
fprintf('调节时间 ts:'); disp(ts);
fprintf('稳态误差 ess:'); disp(ess);

wn_values = [0.2, 2, 20];
sigma_percent = [];
tp = []; tr = []; ts = []; ess = [];
for i = 1:length(wn_values)
    ei = 1.3; wn = wn_values(i); num = wn^2; den = [1, 2*ei*wn, wn^2];
    sys = tf(num, den);
    t = 0:0.01:100;
    [y, t] = step(sys, t); yss = dcgain(sys);
    [ymax, idx_peak] = max(y);
    sigma_percent = [sigma_percent, (ymax - yss) / yss * 100];
    tp = [tp, t(idx_peak)];
    idx_10 = find(y >= 0.1*yss, 1); idx_90 = find(y >= 0.9*yss, 1);
    tr = [tr, t(idx_90) - t(idx_10)];
    idx_settle = find(y < lower_bound | y > upper_bound, 1, 'last');
    if isempty(idx_settle)
        ts = [ts, 0];
    else
        ts = [ts, t(idx_settle)];
    end
    ess = [ess, abs(1 - yss)];
end
fprintf('超调量 σ%%:'); disp(sigma_percent);
fprintf('峰值时间 tp:'); disp(tp);
fprintf('上升时间 tr:'); disp(tr);
fprintf('调节时间 ts:'); disp(ts);
fprintf('稳态误差 ess:'); disp(ess);
