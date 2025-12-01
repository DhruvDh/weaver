use weaver::file_reader::make_conversation_id;

#[test]
fn conversation_ids_use_uuid_v4_and_remain_unique() {
    let actor_name = "FileReader/Root";

    let first = make_conversation_id(actor_name);
    let second = make_conversation_id(actor_name);

    assert!(first.starts_with(actor_name));
    assert!(second.starts_with(actor_name));

    let first_uuid = first.split('#').nth(1).expect("missing uuid segment");
    let second_uuid = second.split('#').nth(1).expect("missing uuid segment");

    let first_parsed = uuid::Uuid::parse_str(first_uuid).expect("invalid uuid format");
    let second_parsed = uuid::Uuid::parse_str(second_uuid).expect("invalid uuid format");

    assert_eq!(first_parsed.get_version_num(), 4);
    assert_eq!(second_parsed.get_version_num(), 4);
    assert_ne!(first_parsed, second_parsed);
}
