clear;
clc;
close all;

omega = 1;
W = 8;

%set up the figure
figure('Renderer', 'painters', 'Position', [10 10 600 500]); hold on;
set(gcf,'color','white'); box on;

Abond = load('data/phase_diagram.csv');
lambda_bond = 4*Abond(:,1).^2/(W*omega);

site_crossover = [0.386363636363636, 0.28551000953288846;
                  0.406471494607087, 0.30753098188751193;
                  0.376656394453004, 0.3328884652049571;
                  0.395377503852080, 0.3635843660629171;
                  0.444607087827426, 0.39995233555767395;
                  0.416872110939907, 0.4436828722376501;
                  0.416178736517719, 0.4446615824594853];

site = [0.24976887519260402,0.17273593898951384;
        0.325346687211094,	0.22311725452812203;
        0.400231124807396,	0.2498093422306959;
        0.475115562403698,	0.25281220209723543;
        0.551386748844376,	0.23512869399428027;
        0.6255778120184899,	0.21043851286939944;
        0.70115562403698,	0.1920877025738799;
        0.7753466872110939,	0.21043851286939944;
        0.850231124807395,	0.16606291706387036;
        0.925115562403698,	0.17373689227836037]

plot(1./lambda_bond,1./Abond(:,2),'ob','MarkerFaceColor','b','MarkerSize',8)
plot(1./site(:,1),site(:,2),'sr','MarkerFaceColor','r','MarkerSize',8)

s0 = csaps(1./lambda_bond,1./Abond(:,2),0.99995); fnplt(s0,'-b');
s0 = csaps(1./site(:,1),site(:,2),0.99); fnplt(s0,'-r');

axis([0,5,0,2.5])
set(gca,'FontName','Times','FontSize',25,'XTick',[0:1:5],'YTick',[0:0.5:4],'linewidth',1)
xlabel('$1/\lambda$','FontSize',25,'Interpreter','latex')
ylabel('$T/t$','FontSize',25,'Interpreter','latex')

legend('CDW (bond-Holstein)','CDW (site-Holstein)')
legend('FontSize',25,'FontName','Times')
legend boxoff;

saveas(gcf,'../figures/phase_diagram.png','png')