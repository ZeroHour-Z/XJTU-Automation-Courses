original = stepinfo(simout1.data, simout1.time);
lead_fix = stepinfo(simout2.data, simout2.time);
lag_fix = stepinfo(simout3.data, simout3.time);
lead_fix1 = stepinfo(simout4.data, simout4.time);
lag_fix1 = stepinfo(simout5.data, simout5.time);


fprintf('原系统:   超调量=%.2f%%, 调节时间=%.2fs\n', original.Overshoot, original.SettlingTime);
fprintf('超前校正: 超调量=%.2f%%, 调节时间=%.2fs\n', lead_fix.Overshoot, lead_fix.SettlingTime);
fprintf('滞后校正: 超调量=%.2f%%, 调节时间=%.2fs\n', lag_fix.Overshoot, lag_fix.SettlingTime);
fprintf('超前校正: 超调量=%.2f%%, 调节时间=%.2fs\n', lead_fix1.Overshoot, lead_fix1.SettlingTime);
fprintf('滞后校正: 超调量=%.2f%%, 调节时间=%.2fs\n', lag_fix1.Overshoot, lag_fix1.SettlingTime);