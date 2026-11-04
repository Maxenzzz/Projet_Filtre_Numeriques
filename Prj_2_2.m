clear;
clc;
close all;

%% Parameters

N = 2000;
R = 0.95;
r = 0.99;
fech = 1;

z1 = R*exp(1j*pi/4);
z2 = conj(z1);

z3 = R*exp(1j*3*pi/4);
z4 = conj(z3);

p1 = r*exp(1j*pi/3);
p2 = conj(p1);

p3 = r*exp(1j*3*pi/7);
p4 = conj(p3);

f_axis = (-N/2: N/2 - 1) * (fech / N);
w_axis = 2 * pi * f_axis / fech;

Kb = 10;
L = N / Kb;

M = 2000;
Nmethod = 200;
Lmethod = floor(M/Nmethod);

Slope = zeros(1, Lmethod);
Offset = zeros(1, Lmethod);
BlocTrendDroite = zeros(Nmethod, Lmethod);
n1 = (1:Nmethod)';

%% Signal Processing

%Filter 1
x = randn(N, 1);

b1 = poly([z1,z2,z3,z4]);
a1 = 1;

h1 = freqz(b1, a1, f_axis, fech);
H_FFT1 = abs(h1);

y1 = filter(b1, a1, x);
y_norm1 = y1/max(abs(y1));

Y1 = fftshift(fft(y1, N));
Y_amp1 = abs(Y1) / N;

power_spec1 = abs(Y1).^2 / (N * fech);
dsp_th1 = (H_FFT1.^2);

power_spec_norm1 = power_spec1 / max(power_spec1);
dsp_th_norm1 = dsp_th1 / max(dsp_th1);


%Filter 2
b2 = 1;
a2 = poly([p1, p2, p3, p4]);

h2 = freqz(b2, a2, f_axis, fech);
H_FFT2 = abs(h2);

y2 = filter(b2, a2, x);
y_norm2 = y2 / max(abs(y2));

Y2 = fftshift(fft(y2, N));
Y_amp2 = abs(Y2) / N;

power_spec2 = abs(Y2).^2 / (N * fech);
dsp_th2 = H_FFT2.^2;

power_spec_norm2 = power_spec2 / max(power_spec2);
dsp_th_norm2 = dsp_th2 / max(dsp_th2);

% Periodograms

PDaniell1 = pDaniell(y1, 4);
PDaniell2 = pDaniell(y2, 4);

[PBartlett1, f1] = pwelch(y1, rectwin(L), 0, N, fech, "centered");
[PBartlett2, f2] = pwelch(y2, rectwin(L), 0, N, fech, "centered");

[PWelch1, f1_w] = pwelch(y1, hann(L), 40, N, fech, "centered");
[PWelch2, f2_w] = pwelch(y2, hann(L), 40, N, fech, "centered");

% Correlograms

[Corr1, lag_axis1] = xcorr(Y1, "biased");
[Corr2, lag_axis2] = xcorr(Y2, "biased");

lag_axis1 = lag_axis1 / (2*N);
lag_axis2 = lag_axis2 / (2*N);

%% Method

% Step 1

x2 = randn(1,M);
x_center = x2-mean(x2);
y3 = zeros(1,M);

y3(1) = x_center(1);
for n=2:M
    y3(n) = x_center(n)+y3(n-1);
end

% Step 2
% 2a 

yTronq = y3(1:Lmethod*Nmethod);
yBloc = reshape(yTronq, Nmethod, Lmethod);

% Mean

TrendMean = zeros(1,Lmethod);
BlocTrendMean = zeros(Nmethod,Lmethod);

for i=1:Lmethod
    TrendMean(i) = mean(yBloc(:,i));
    BlocTrendMean(:,i) = TrendMean(i);
end
GlobalTrendMean = reshape(BlocTrendMean, 1, []);

% Droite

for i = 1:Lmethod
    Bloc1 = yBloc(:,i);
    Slope(i) = sum((n1-mean(n1)).*(Bloc1-mean(Bloc1))) / sum((n1-mean(n1)).^2);
    Offset(i) = mean(Bloc1) - Slope(i)*mean(n1);
    BlocTrendDroite(:,i) = Slope(i)*n1 + Offset(i);
end
GlobalTrendDroite = reshape(BlocTrendDroite,1,[]);

% Polynome 2nd deg

A = [n1.^2, n1, ones(Nmethod, 1)];

for j = 1:Lmethod
    Bloc1 = yBloc(:,j);
    theta = inv(A'*A)*A'*Bloc1;
    a = theta(1);
    b = theta(2);
    c = theta(3);
    BlocTrendPoly2(:,j) = a*n1.^2 + b*n1 + c;
end
GlobalTrendPoly2 = reshape(BlocTrendPoly2,1,[]);

%Polynome du 3e deg

B = [n1.^3, n1.^2, n1, ones(Nmethod, 1)];

for j = 1:Lmethod
    Bloc1 = yBloc(:,j);
    theta = inv(B'*B)*B'*Bloc1;
    a = theta(1);
    b = theta(2);
    c = theta(3);
    d = theta(4);
    BlocTrendPoly3(:,j) = a*n1.^3 + b*n1.^2 + c*n1 + d;
end
GlobalTrendPoly3 = reshape(BlocTrendPoly3,1,[]);

% 2b
% Filtrage PB, moyenne glissante

bMean = ones(1,Nmethod)/Nmethod;
aMean = 1;
GlobalTrendFiltre = filter(bMean,aMean,y3);

% 2c
% Stratégie 1
% PB

h_PB = ones(1, Nmethod)/Nmethod;
g_PB = filter(h_PB, 1, y3); 
gf_PB = flip(g_PB);

r_PB = filter(h_PB, 1, gf_PB);
rf_PB = flip(r_PB);

% Passe Haut

h_PH = fir1(Nmethod, 0.01, 'high');
g_PH = filter(h_PH, 1, y3); 
gf_PH = flip(g_PH);

r_PH = filter(h_PH, 1, gf_PH);
rf_PH = flip(r_PH);

% Passe Bande

h_PBd = fir1(Nmethod, [0.01 0.1]);
g_PBd = filter(h_PBd, 1, y3); 
gf_PBd = flip(g_PBd);

r_PBd = filter(h_PBd, 1, gf_PBd);
rf_PBd = flip(r_PBd);

% Coupe Bande

h_CB = fir1(Nmethod, [0.01 0.1], 'stop');
g_CB = filter(h_CB, 1, y3); 
gf_CB = flip(g_CB);

r_CB = filter(h_CB, 1, gf_CB);
rf_CB = flip(r_CB);

%% Plotting

zplane(b1, a1);
figure;
subplot(1,2,1);
plot(f_axis, 10*log10(Y_amp1))
xlabel("Axe fréquentiel");
ylabel("Amplitude normalisé");
title("Spectres d'amplitude du flitres RIF");
grid on;

subplot(1,2,2);
plot(f_axis, 10*log10(Y_amp2));
xlabel("Axe fréquentiel");
ylabel("Amplitude normalisée");
title("Spectres d'amplitude des filtres RII");
grid on;

figure;
plot(f_axis, 10*log10(power_spec_norm1));
hold on;
plot(f_axis, 10*log10(dsp_th_norm1), "r--");
xlabel("Axe fréquentiel");
ylabel("Puissance normalisée (dB)");
title("Densité spectrale de puissance théorique et empirique");
grid on;
hold off;

figure;
plot(f_axis, 10*log10(power_spec_norm2));
hold on;
plot(f_axis, 10*log10(dsp_th_norm2), "r--");
xlabel("Axe fréquentiel");
ylabel("Puissance normalisée (dB)");
title("Densité spectrale de puissance théorique et empirique");
grid on;
hold off;

figure;
plot(f_axis, 10*log10(dsp_th_norm1));
hold on;
plot(f_axis, 10*log10(PDaniell1 / max(PDaniell1)), "r--");
xlabel("Axe fréquentiel");
ylabel("Périodogramme de Daniell normalisé (dB)");
title("Estimations spectrales lissées par un periodograme de Daniell");
legend("DSP Filtre1", "PDaniell Filtre 1");
grid on;

figure;
plot(f_axis, 10*log10(dsp_th_norm2));
hold on;
plot(f_axis, 10*log10(PDaniell2 / max(PDaniell2)), "r--");
xlabel("Axe fréquentiel");
ylabel("Périodogramme de Daniell normalisé (dB)");
title("Estimations spectrales lissées par un periodograme de Daniell");
legend("DSP Filtre2", "PDaniell Filtre 2");
grid on;

figure;
plot(f_axis, 10*log10(dsp_th_norm1));
hold on;
plot(f1, 10*log10(PBartlett1 / max(PBartlett1)), "r--");
xlabel("Axe fréquentiel");
ylabel("Périodogramme de Bartlett normalisé (dB)");
title("Estimations spectrales lissées par un periodograme de Bartlett");
legend("DSP Filtre2", "PBartlett Filtre 2");
grid on;

figure;
plot(f_axis, 10*log10(dsp_th_norm2));
hold on;
plot(f2, 10*log10(PBartlett2 / max(PBartlett2)), "r--");
xlabel("Axe fréquentiel");
ylabel("Périodogramme de Bartlett normalisé (dB)");
title("Estimations spectrales lissées par un periodograme de Bartlett");
legend("DSP Filtre2", "PBartlett Filtre 2");
grid on;

figure;
plot(f_axis, 10*log10(dsp_th_norm1));
hold on;
plot(f1_w, 10*log10(PWelch1 / max(PWelch1)), "r--");
xlabel("Axe fréquentiel");
ylabel("Périodogramme de Welch normalisé (dB)");
title("Estimations spectrales lissées par un periodograme de Welch");
legend("DSP Filtre 1", "PWelch Filtre 1");
grid on;

figure;
plot(f_axis, 10*log10(dsp_th_norm2));
hold on;
plot(f2_w, 10*log10(PWelch2 / max(PWelch2)), "r--");
xlabel("Axe fréquentiel");
ylabel("Périodogramme de Welch normalisé (dB)");
title("Estimations spectrales lissées par un periodograme de Welch");
legend("DSP Filtre 2", "PWelch Filtre 2");
grid on;

figure;
plot(f_axis, 10*log10(dsp_th_norm1));
hold on;
plot(lag_axis1, 10*log10(real(Corr1) / max(real(Corr1))), "r--");
xlabel("Axe fréquentiel");
ylabel("Périodogramme de Welch normalisé (dB)");
title("Correlation spectral de Y1");
legend("DSP Filtre 1", "Auto correlation de Y1");
grid on;

figure;
plot(f_axis, 10*log10(dsp_th_norm2));
hold on;
plot(lag_axis2, 10*log10(real(Corr2)/max(real(Corr2))), "r--");
xlabel("Axe fréquentiel");
ylabel("Périodogramme de Welch normalisé (dB)");
title("Correlation spectral de Y2");
legend("DSP Filtre 2", "Auto correlation de Y2");
grid on;

figure;
plot(yTronq, 'k');
hold on;
plot(GlobalTrendMean, 'b', 'LineWidth', 1.5);
plot(GlobalTrendDroite, 'r', 'LineWidth', 1.5);
plot(GlobalTrendPoly2, 'g', 'LineWidth', 1.5);
plot(GlobalTrendPoly3, 'Color', '#D95319', 'LineWidth', 1.5);
grid on;
legend('Signal', 'Moyenne', 'Regression lineaire', 'Polynome 2nd deg', 'Polynome 3e deg');

figure;
plot(y3,'k');
hold on;
plot(GlobalTrendFiltre,'m','LineWidth',1.5);
grid on;
title('Signal filtré par un filtre passe bas');

figure;
plot(y3, 'k');
hold on;
plot(rf_PB, 'b', 'LineWidth', 1.5);
plot(rf_PH, 'r', 'LineWidth', 1.5);
plot(rf_PBd, 'g', 'LineWidth', 1.5);
plot(rf_CB, 'Color', '#D95319', 'LineWidth', 1.5);
legend('Signal', 'Filtré Passe Bas', 'Filtré Passe Haut', 'Filtré Passe Bande', 'Filtré Coupe Bande');
grid on;
title('Stratégie 1 avec différents filtres');

xlabel('Echantillons');
ylabel('Amplitude');
title('Estimation par moyenne glissante');
legend('Signal initial','Tendance filtree');

function P = pDaniell(x, w)
    X = abs(fftshift(fft(x))).^2;
    P = movmean(X, w).';
end