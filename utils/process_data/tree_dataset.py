import pandas as pd

"""
class Topic:

    def __init__(self, topic_id):
        self.id = topic_id

    @property
    def parent(self):
        parent_id = topics.loc[self.id].parent
        if pd.isna(parent_id):
            return None
        else:
            return Topic(parent_id)

    @property
    def ancestors(self):
        ancestors = []
        parent = self.parent
        while parent is not None:
            ancestors.append(parent)
            parent = parent.parent
        return ancestors

    @property
    def siblings(self):
        if not self.parent:
            return []
        else:
            return [topic for topic in self.parent.children if topic != self]

    def get_breadcrumbs(self, separator=" >> ", include_self=True, include_root=True):
        # Gather breadcrumbs
        ancestors = self.ancestors
        if include_self:
            ancestors = [self] + ancestors
        if not include_root:
            ancestors = ancestors[:-1]
        if not ancestors:
            return ''
        res = [ancestors[0].title]
        for a in ancestors[1:]:
            res.append(a.title)
        res = list(reversed(res))
        return separator.join(res[1:] + [res[0]])

    @property
    def children(self):
        return [Topic(child_id) for child_id in topics[topics.parent == self.id].index]

    def get_children_titles(self):
        children_titles = [Topic(child_id).title for child_id in topics[topics.parent == self.id].index]
        children_titles = np.unique(children_titles)
        children_titles = ' '.join(children_titles)
        if not children_titles:
            return 'no children'
        return children_titles

    def __eq__(self, other):
        if not isinstance(other, Topic):
            return False
        return self.id == other.id

    def __getattr__(self, name):
        return topics.loc[self.id][name]

    def __str__(self):
        return self.title


topics, contents, ss = load_datasets(INPUT_DIR)

# fillna
topics['title'] = topics['title'].fillna('no title')
topics = topics.set_index('id')

breadcrumbs = [Topic(t).get_breadcrumbs(include_self=False) for t in tqdm(topics.index)]
breadcrumbs = pd.DataFrame(breadcrumbs, columns=['context'])
breadcrumbs['id'] = topics.index

# children
children = [Topic(t).get_children_titles() for t in tqdm(topics.index)]
children = pd.DataFrame(children, columns=['children_title'])
children['id'] = topics.index

del Topic
gc.collect()

# cousines
topics['cousine'] = topics['parent'].map(topics.groupby('parent')['title'].apply(lambda x: ' '.join(np.unique(x))))
topics['cousine'] = topics['cousine'].fillna('no cousine')
topics['cousine'] = topics.apply(lambda row: row.cousine.replace(row.title, ''), axis=1)
topics['cousine'] = topics['cousine'].replace('', 'no cousine')

# reset index
topics = topics.reset_index()

# get topics for inference
topics_test = get_inference_topics(topics, ss)
del topics, ss
gc.collect()



"""