import re


def parse_intake_forms(data):
    entire_intake_forms = []
    for _, row in data.iterrows():
        sections = re.split("\n[\s\t]*\n", row['intake_form'])
        parsed_data = {'personal_info': {}}
        
        personal_info_str = sections[0].replace(":\n", ": ").split("\n")
        for line in personal_info_str:
            k, v = line.split(":", 1)
            parsed_data['personal_info'][f'{k.lower()}'] = v.strip()

        for section in sections[1:]:
            try:
                title, _, content = re.split('(\n|\:)+', section, maxsplit=1)
                title = re.sub(r'^\d+\.\s*', '', title.lower().strip().rstrip(':'))
                parsed_data[title] = content.strip()
            except ValueError:
                print(section)
                pass
        entire_intake_forms.append(parsed_data)
    
    data['reason_counseling'] = [x.get('reason for seeking counseling', '') for x in entire_intake_forms]
    data['personal_info'] = [str(x.get('personal_info', {})) for x in entire_intake_forms]
    data['presenting problem'] = [x.get('presenting problem', '') for x in entire_intake_forms]
    
    return data