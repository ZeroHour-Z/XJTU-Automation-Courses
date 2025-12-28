num1=1;
den11=[0.5 1];
den12=[1 1]; 
den13=[2 1]; 
G1_1=tf(num1, den11);
G1_2=tf(num1, den12);
G1_3=tf(num1, den13);
bode(G1_1, G1_2, G1_3); grid on;
figure;
step(feedback(G1_1,1),feedback(G1_2,1),feedback(G1_3,1));grid on;
title('惯性环节');

num21=[0.5 1];
num22=[1 1];
num23=[2 1];
den2=[1 1];
G2_1=tf(num21, den2);
G2_2=tf(num22, den2);
G2_3=tf(num23, den2);
figure
bode(G2_1, G2_2, G2_3); grid on;
title('微分环节');

num31=0.5;
num32=1;
num33=2;
den3=1;
G3_1=tf(num31, den3);
G3_2=tf(num32, den3);
G3_3=tf(num33, den3);
figure
bode(G3_1, G3_2, G3_3); grid on;
title('比例环节');

num4=[1];
den41=[0.5 0];
den42=[1 0];
den43=[2 0];
G4_1=tf(num4, den41);
G4_2=tf(num4, den42);
G4_3=tf(num4, den43);
figure
bode(G4_1, G4_2, G4_3); grid on;
title('积分环节');

G5_1=0.5+tf(num4, den41);
G5_2=0.5+tf(num4, den43);
G5_3=2+tf(num4, den41);
G5_4=2+tf(num4, den43);
figure
bode(G5_1, G5_2, G5_3, G5_4); grid on;
title('比例积分');

G6_1=0.5+tf(num21, den2);
G6_2=0.5+tf(num23, den2);
G6_3=2+tf(num21, den2);
G6_4=2+tf(num23, den2);
figure
bode(G6_1, G6_2, G6_3, G6_4); grid on;
title('比例微分');