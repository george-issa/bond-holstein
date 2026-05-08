clear;
clc;
close all;

A = load('data/phase_diagram.csv');
W = 8;
Omega = 1;
lambdaA = (8*A(:,1).^2)/(W*Omega);

B = load('data/CDW_betacs.csv');

figure('Renderer', 'painters', 'Position', [10 10 500 500])
set(gcf,'color','white'); hold on;

errorbar(1./lambdaA,A(:,2),A(:,3),'ok','MarkerFaceColor','k','MarkerSize',10)
