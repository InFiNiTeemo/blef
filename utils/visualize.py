def visualize_model(model):
    ...
    # model = cfg.MyModel(args.model_name_or_path, logger=logger, dropout_rate=cfg.dropout_rate, pooler=cfg.pooler).cuda()
    # todo visualize
    # if fold == 0:
    #     t_input, t_mask = torch.zeros(5,3).long().cuda(), torch.ones(5,3).long().cuda()
    #     # output = model(t_input, t_mask).cpu()
    #
    #     #with SummaryWriter(output_path, comment="sample_model_visualization") as sw:
    #     #    sw.add_graph(model, [t_input, t_mask])
    #
    #     import hiddenlayer as hl
    #
    #     transforms = [hl.transforms.Prune('Constant')]  # Removes Constant nodes from graph.
    #
    #     graph = hl.build_graph(model, [t_input, t_mask], transforms=transforms)
    #     graph.theme = hl.graph.THEMES['blue'].copy()
    #     graph.save(os.path.join(output_path, "pic"), format='png')
    #     # make_dot(output, params=dict(list(model.named_parameters()))).render(output_path, format="png")
    #     # torchvz not work
    #     # g = make_dot(output, params=dict(model.named_parameters()))
    #     # g.format = "png"
    #     # g.directory = output_path
    #     # g.view()
    #     # g.view(os.path.join(output_path, "model_view"))
    #     #g.render(os.path.join(output_path, "model_view"), format="png", view=False)
