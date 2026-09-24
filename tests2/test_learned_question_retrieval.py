import unittest
from legalir2.learned_question_retrieval import LearnedQuestions


class TestLearnedQuestions(unittest.TestCase):
    def records(self):
        return {'a':{'question':'thuế thu nhập cá nhân tiền lương','answer':['tax']},
                'b':{'question':'thuế cá nhân thu nhập chuyển nhượng','answer':['tax']},
                'c':{'question':'đăng ký khai sinh trẻ em hộ tịch','answer':['birth']}}

    def test_training_only_label_map_and_exclusion(self):
        model=LearnedQuestions(self.records(),{'tax','birth'})
        rows=model.propose('thuế thu nhập cá nhân miễn giảm',[],qid='test')
        self.assertEqual(rows[0]['doc'],'tax')
        self.assertNotIn('tax',[r['doc'] for r in model.propose('thuế cá nhân',['tax'])])

    def test_leakage_and_unknown_labels_fail(self):
        with self.assertRaisesRegex(ValueError,'overlaps'):
            LearnedQuestions(self.records(),{'tax','birth'},excluded_ids=['a'])
        with self.assertRaisesRegex(ValueError,'overlaps'):
            LearnedQuestions(self.records(),{'tax','birth'},excluded_texts=['THUẾ THU NHẬP CÁ NHÂN TIỀN LƯƠNG?'])
        with self.assertRaisesRegex(ValueError,'labels'):
            LearnedQuestions(self.records(),{'tax'})

    def test_training_self_prediction_is_refused(self):
        model=LearnedQuestions(self.records(),{'tax','birth'})
        with self.assertRaisesRegex(ValueError,'excluded fold'):
            model.scores('arbitrary query',qid='a')
        with self.assertRaisesRegex(ValueError,'excluded fold'):
            model.scores(self.records()['b']['question'])
        self.assertEqual(model.propose('xyzzyqqq',[]),[])

    def test_duplicate_text_cannot_amplify_labels_or_choose_conflicting_label(self):
        records=self.records()
        records['a2']=dict(records['a'])
        model=LearnedQuestions(records,{'tax','birth'})
        self.assertEqual(model.cleaning['identical_duplicates_removed'],1)
        records['c2']={'question':records['c']['question'],'answer':['tax']}
        model=LearnedQuestions(records,{'tax','birth'})
        self.assertEqual(model.cleaning['conflicting_rows_excluded'],2)
        self.assertNotIn('birth',model.documents)


if __name__=='__main__':
    unittest.main()
