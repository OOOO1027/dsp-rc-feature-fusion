
function plot_hist3nan(dataXlms)
edge = max(max(abs(real(dataXlms))),max(abs(imag(dataXlms))));
% edge = edge + 0.05*edge;
edges = linspace(-edge,edge,250);
z = hist3([real(dataXlms);imag(dataXlms)].','Edges',{edges,edges});
%  z = max(max(z))-z;
figure
surface(edges,edges,z)
% grid on
shading interp
load constellationcolor.txt;
colormap(constellationcolor)

% colormap jet
a=max(abs(real(dataXlms)));
b=max(abs(imag(dataXlms)));
axis equal
axis([-a-0.5,a+0.5,-b-0.5,b+0.5]);
% color('white');

end