    # ap.add_argument("-r", "--raio", default=False, 
    #                     help="Raio a ser utilizado na geração do pré-processamento LBP")
    
    # ap.add_argument("-n", "--numeroPontos", default=False, 
    #                     help="Número de pontos a ser utilizado na geração do pré-processamento LBP")
    
    # ap.add_argument("-grid", "--grid", default=False, 
    #                     help="Tamanho do grid a ser usado no LBPH, no formato ex.: 8x8")
    

    # if args.grid:
    #     try:
    #         grid = args.grid.split('x')
    #         w = size[0]
    #         s = size[1]
    #     except:
    #         logging.error('Não foi possível ler o tamanho no formato: %s' % args.grid)
    #         sys.exit()

    # def concatenaLBP(path1, path2, numPoints, radius, method):
    # _, hist1 = aplica_lbp(path1, numPoints, radius, method)
    # _, hist2 = aplica_lbp(path2, numPoints, radius, method)
    # return np.concatenate((hist1, hist2))

    #     posfix = []
    # if descricao_dict is not None:
    #     for k,v in descricao_dict.items():
    #         posfix += '_'.join[str(k), str(descricao_dict[k])]