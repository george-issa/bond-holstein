clear;
clc;
close all;

Data8  = load('data/CN8new.dat');
Data12 = load('data/CN12new.dat');
Data16 = load('data/CN16new.dat');

Binder8  = load('data/BinderN8new.dat');
Binder12 = load('data/BinderN12new.dat');
Binder16 = load('data/BinderN16new.dat');

figure('Renderer', 'painters', 'Position', [10 10 500 700])
set(gcf,'color','white')

Left = 0.15;
Bottom = 0.1;
Height = 0.39;
Width = 0.8;
voffset = 0.1;
T0 = 1.76;

C = [0:0.1:5];

subplot('position',[Left,Bottom+Height+voffset,Width,Height]); 
box on; hold on;
set(gca,'FontSize',20,'FontName','Times','LineWidth',1)
xlabel('$T/|U_\mathrm{eff}|$','FontSize',20,'Interpreter','latex')
ylabel('Specific Heat, $C$','FontSize',20,'Interpreter','latex')
plot(Data8(:,1),Data8(:,2),'-ro','MarkerFaceColor','r','MarkerSize',8,'LineWidth',1)
plot(Data12(:,1),Data12(:,2),'-bv','MarkerFaceColor','b','MarkerSize',8,'LineWidth',1)
plot(Data16(:,1),Data16(:,2),'-gs','MarkerFaceColor','g','MarkerSize',8,'LineWidth',1)
plot(T0*ones(size(C)),C,'--k','HandleVisibility','off','LineWidth',1)
axis([1.25,3,0,4])
legend('$8\times 8$', '$12\times 12$','$16\times16$')
legend('location','northeast','Interpreter','latex')
legend boxoff;
text(1.25*1.05,4*0.92,'(a)','FontSize',20,'FontName','Times')

subplot('position',[Left,Bottom,Width,Height]); 
box on; hold on;
set(gca,'FontSize',20,'FontName','Times','LineWidth',1)
xlabel('$T/|U_\mathrm{eff}|$','FontSize',20,'Interpreter','latex')
ylabel('Binder ratio, $B$','FontSize',20,'Interpreter','latex')
plot(Binder8(:,1),Binder8(:,2),'-ro','MarkerFaceColor','r','MarkerSize',8,'LineWidth',1)
plot(Binder12(:,1),Binder12(:,2),'-bv','MarkerFaceColor','b','MarkerSize',8,'LineWidth',1)
plot(Binder16(:,1),Binder16(:,2),'-gs','MarkerFaceColor','g','MarkerSize',8,'LineWidth',1)
plot(T0*ones(size(C)),C,'--k','HandleVisibility','off')
axis([1.25,3,0,0.8])
text(1.25*1.05,0.8*0.92,'(b)','FontSize',20,'FontName','Times','LineWidth',1)

axes('Position',[.63 .30 .3 .15],'LineWidth',1); hold on; box on;
plot(Binder8(:,1),Binder8(:,2),'-ro','MarkerFaceColor','r','MarkerSize',8,'LineWidth',1)
plot(Binder12(:,1),Binder12(:,2),'-bv','MarkerFaceColor','b','MarkerSize',8,'LineWidth',1)
plot(Binder16(:,1),Binder16(:,2),'-gs','MarkerFaceColor','g','MarkerSize',8,'LineWidth',1)
axis([1.71,1.82,0.55,0.65])
set(gca,'FontSize',18,'FontName','Times','LineWidth',1,'XTick',[1.72:0.02:1.82])
xtickangle(0)
xticklabels({'1.72',' ','1.76',' ','1.80',' '})
plot(T0*ones(size(C)),C,'--k','HandleVisibility','off','LineWidth',1)

saveas(gcf,'../figures/teq0.pdf','pdf')
