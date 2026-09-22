"""Metadata ownership and bounded citation proposals without labels."""
import unittest
from legalir2.local_authorities import Authorities, reference_numbers


class TestAuthorities(unittest.TestCase):
    def fixture(self):
        return Authorities({'anchor':{'title':'example'},'law':{'title':'Bo luat dan su 2015 296215'},
            'letter':{'number':'1523/BGDĐT-QLCL'},'decree':{'number':'007/2021/NĐ-CP'}})

    def test_yearless_numbers_and_named_laws(self):
        model=self.fixture()
        found=model.references('Theo Bộ luật Dân sự năm 2015, Công văn 1523/BGDĐT-QLCL và Nghị định 7/2021/NĐ-CP.')
        self.assertEqual({r[0] for r in found},{'law','letter','decree'})

    def test_ambiguous_names_never_choose_arbitrary_version(self):
        model=Authorities({'old':{'title':'Luat dat dai 2003 123456'},'new':{'title':'Luat dat dai 2013 234567'}})
        self.assertEqual(model.references('Luật đất đai'),[])
        self.assertEqual([r[0] for r in model.references('Luật đất đai 2013')],['new'])
        self.assertEqual([r[0] for r in model.references('Luật đất đai năm 2013')],['new'])
        self.assertEqual(model.references('Luật đất đai năm 2024'),[])

    def test_explicit_wrong_version_never_falls_back_to_unique_name(self):
        self.assertEqual(self.fixture().references('Bộ luật dân sự năm 2005'),[])
        self.assertEqual(self.fixture().references('Bộ luật dân sự sửa đổi 2020'),[])

    def test_number_issuer_accepts_digits(self):
        model=Authorities({'law':{'number':'69/2022/QH15'}})
        self.assertEqual([r[0] for r in model.references('Nghị quyết 69/2022/QH15')],['law'])
        self.assertEqual(model.references('Nghị quyết 69/2022/QH14'),[])

    def test_longer_ambiguous_name_blocks_short_unique_name(self):
        model=Authorities({'short':{'title':'Luat bao ve'},'a':{'title':'Luat bao ve moi truong 2014'},
                           'b':{'title':'Luat bao ve moi truong 2020'}})
        self.assertEqual(model.references('Luật bảo vệ môi trường'),[])

    def test_amendments_cannot_own_base_law_alias(self):
        model=Authorities({'a':{'title':'Luat dan su sua doi 2020 123456'}})
        self.assertEqual(model.references('Luật dân sự'),[])
        model=Authorities({'a':{'title':'Luat dan su 2015 sua doi 2020 123456'}})
        self.assertEqual(model.references('Luật dân sự 2015'),[])

    def test_existing_candidates_excluded_and_repeated_cites_not_double_counted(self):
        model=self.fixture()
        traces={'anchor':{'packed':[{'text':'Bộ luật dân sự 2015. Bộ luật dân sự 2015.'}]}}
        proposal=model.propose('dân sự',['anchor'],['anchor'],traces)
        self.assertEqual([p['doc'] for p in proposal],['law'])
        self.assertEqual(len(proposal[0]['reasons']),1)
        self.assertEqual(model.propose('dân sự',['anchor'],['anchor','law'],traces),[])

    def test_proposals_are_bounded_and_ignore_fifth_answer(self):
        documents={'anchor':{},'fifth':{}}
        documents.update({str(i):{'number':f'{i}/2020/ND-CP'} for i in range(1,7)})
        model=Authorities(documents)
        text=' '.join(f'{i}/2020/ND-CP' for i in range(1,7))
        traces={'anchor':{'packed':[{'text':text}]},'fifth':{'packed':[{'text':text}]}}
        self.assertEqual(len(model.propose('test',['anchor'],['anchor'],traces)),4)
        self.assertEqual(len(model.candidates('test',['anchor'],['anchor'],traces)),6)
        for doc in ('a','b','c','d'):
            documents[doc]={}
        model=Authorities(documents)
        self.assertEqual(model.propose('test',['a','b','c','d','fifth'],['fifth'],traces),[])

    def test_unknown_corpus_identity_fails(self):
        with self.assertRaises(ValueError):
            self.fixture().propose('test',['unknown'],[],{})


if __name__=='__main__':
    unittest.main()
