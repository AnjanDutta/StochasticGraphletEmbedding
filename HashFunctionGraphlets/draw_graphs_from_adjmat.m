function h = draw_graphs_from_adjmat(A)

N = size(A,1);

vertices = [cos(2*pi*(1:N)/N); sin(2*pi*(1:N)/N)]';

[x,y] = gplot(A, vertices);
h = figure; hold on;
plot(x, y, '-ok', 'LineWidth', 3, 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b', 'MarkerSize', 15 );
hold off;
axis off;

end