import torch
import numpy as np
import faiss


# 验证/测试中获取图片经模型处理后的features
def generate_test_features(model, test_loader, mode="test"):
    model.eval()  # 模型调整到评估模式
    FEAS = []
    IDS = []
    TARGETS = []
    if mode == "vec":
        with torch.no_grad():
            # for batch_idx, (images, posting_id) in enumerate(test_loader): # 从数据管道中导入 图片
            #    posting_id = tenjsor.LongTensor(1)
            for batch_idx, d in enumerate(test_loader):  # 从数据管道中导入 图片
                vec = model.generate_vec("ctx", d["c_ids"].cuda(), d["c_mask"].cuda(), True)
                # image, posting_id = images
                IDS.append(d["raw_c_id"])
                FEAS += [vec.detach().cpu()]  # 存下当前features
                # IDS += [posting_id.detach().cpu()]
        FEAS = torch.cat(FEAS).cpu().numpy()
        IDS = np.concatenate(IDS, axis=0)
        # IDS = torch.cat(IDS).numpy().reshape(-1)
        return FEAS, IDS  # 返回所有数据的features

def pipeline():
    vec_loader = None
    FEAS, IDS = generate_test_features(vec_loader, mode="vec")
    FEAS.shape, IDS.shape
    np.save(f'a/img_feats.npy', FEAS)


def query():
    import gc
    import numpy as np
    import faiss

    def query_expansion(feats, sims, topk_idx, alpha=0.5, k=2):
        weights = np.expand_dims(sims[:, :k] ** alpha, axis=-1).astype(np.float32)
        feats = (feats[topk_idx[:, :k]] * weights).sum(axis=1)
        return feats

    img_feats = np.load(f'a/img_feats.npy')

    res = faiss.StandardGpuResources()
    index_img = faiss.IndexFlatIP(img_feats.shape[1])  # 初始化向量维度d,
    index_img = faiss.index_cpu_to_gpu(res, 0, index_img)  #
    index_img.add(img_feats)
    img_D, img_I = index_img.search(img_feats, 60) # 搜索最近的n个向量

    # groups = df_train.label_group.values

class FaissEngine():
    def __init__(self, metrics="cosine", gpu=False):
        # https://zhuanlan.zhihu.com/p/90768014
        # faiss 最常用的三种索引 IndexFlatL2, IndexIVFFlat(倒排， Kmeans聚类),IndexIVFPQ。
        self.faiss = None
        self.resource = None
        self.use_gpu = gpu
        self.metrics = metrics

    def init_engine(self, FEAS, IDS):
        print("\nStart adding vecs.")
        # FEAS, IDS = generate_test_features(model, data_loader, "vec")

        print("vec shape:", FEAS.shape)

        res = faiss.StandardGpuResources()
        self.resource = res

        # print("save vec: ", np.linalg.norm(FEAS, axis=1))
        # cosine == inner procduct != L2, 两个模为1的向量L2最大为2, 而cosine最大为1
        if self.metrics == "cosine":
            index = faiss.IndexFlatIP(FEAS.shape[1])  # IP means inner product
        elif self.metrics == "l2":
            index = faiss.IndexFlatL2(FEAS.shape[1])  # 初始化向量维度d,
        else:
            raise NotImplementedError("Not Implement init engine.")
        index = faiss.IndexIDMap(index)  # 由于FlatL2不支持add with ids
        index.add_with_ids(FEAS, IDS)
        if self.use_gpu:
            index = faiss.index_cpu_to_gpu(res, 0, index)  # oom for all-MiniLM-L12-v2

        print("End of adding vecs.\n")
        self.faiss = index
        # img_D, img_I = index_img.search(FEAS, 60)  # 搜索最近的n个向量

    def search(self, vec, num=100):
        # print("search vec: ", np.linalg.norm(vec, axis=1))
        score, idx = self.faiss.search(vec, num)
        return score, idx

    def reset(self):
        # print("Release faiss memory...")
        self.faiss.reset()
        # print("after reset: {}".format(self.resource.noTempMemory()))  # will output None