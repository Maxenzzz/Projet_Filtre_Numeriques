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
x_center = x-mean(x);
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
plot(yTronq);
hold on;
plot(GlobalTrendMean, 'r', 'LineWidth', 2);
legend('Signal', 'Tendance estimée');
grid on;

function P = pDaniell(x, w)
    X = abs(fftshift(fft(x))).^2;
    P = movmean(X, w).';
end