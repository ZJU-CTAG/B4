from collections import defaultdict

from src.io_utils import Tools
import json

STOP_TOKEN = ['\nclass', '\ndef', '\n#', '\nif', '\nprint']

class PostProcessor:
    @staticmethod
    def map_task_id_for_solution(predict_path, source_path):
        raw_problems = Tools.load_tasks(source_path)

        result = []
        predictions = json.load(open(predict_path))
        for task_id in predictions:
            task = raw_problems[task_id]
            pre = predictions[task_id]
            # if not pre['samples']:
            #     result.append({
            #         'task_id': task['task_id'],
            #         'prompt': pre['prompt'],
            #         'test': task['test'],
            #         'entry_point': task['entry_point'],
            #         'completion': 'empty solution here, execution will fail'
            #     })
            for sample in pre:
                if "Deepseek" in predict_path:
                    result.append({
                        'task_id': task['task_id'],
                        'prompt': '',
                        'test': task['test'],
                        'entry_point': task['entry_point'],
                        'completion': sample
                    })
                else:
                    result.append({
                        'task_id': task['task_id'],
                        'prompt': task['prompt'],
                        'test': task['test'],
                        'entry_point': task['entry_point'],
                        'completion': sample
                    })

        return result

    @staticmethod
    def map_task_id_for_test_case(predict_path, source_path):
        return json.load(open(predict_path))

    @staticmethod
    def solution_extract(content):
        for identifier in STOP_TOKEN:
            if identifier in content:
                content = content.split(identifier)[0]
        return content

    @staticmethod
    def solution_extract_deepseek(content):
        stop_words = ['\n<|EOT|>', "\nclass", "\nif", "\n#", "\nprint"]
        for identifier in stop_words:
            if identifier in content:
                content = content.split(identifier)[0]
        return content    

    @staticmethod
    def solution_extract_wizardcoder(content):
        stop_words = ['\n<|EOT|>', "\nclass", "\nif", "\n#", "\nprint"]
        for identifier in stop_words:
            if identifier in content:
                content = content.split(identifier)[0]
        return content    

    @staticmethod
    def solution_extract_magicoder(content):
        stop_words = ['\n<|EOT|>', "\nclass", "\nif", "\n#", "\nprint",'```']
        for identifier in stop_words:
            if identifier in content:
                content = content.split(identifier)[0]
        return content    
    
    
    @staticmethod
    def test_case_extract(content, entry_point):
        def _truncate(content):
            for identifier in STOP_TOKEN:
                if identifier in content:
                    content = content.split(identifier)[0]
            return content.strip()
        
        split_by_assert = [f'assert {part}'.strip() for part in f'assert {content}'.split('assert ') if (entry_point.strip() in part) and len(part.strip()) > 0]
        truncated_test_cases = [_truncate(i) for i in split_by_assert]
        checked_assertions = [i for i in truncated_test_cases if PostProcessor._check_test_case_validation(i)]
        return checked_assertions

    @staticmethod
    def _check_test_case_validation(test_case):
        if len(test_case.strip()) < 1:
            return False
        if 'assert' not in test_case:
            return False
        try:
            multi_line_test_case = test_case.replace("\n", "\n    ")
            assert_in_a_block = f'try:\n    {multi_line_test_case}\nexcept:\n    pass\n'
            compile(assert_in_a_block, '', 'exec')
            return True
        except Exception:
            return False